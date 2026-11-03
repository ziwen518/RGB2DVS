"""Frozen PureSpikeFormer downstream probes for DSEC.

The encoder is never updated here.  Detection and semantic-segmentation labels
are consumed only by their respective downstream heads.  This keeps the
representation objective label-free while providing a repeatable benchmark
probe for checkpoints saved during representation training.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from data import DSECEvents  # noqa: E402
from model import PureSpikeFormer  # noqa: E402


DET_CLASSES = 3  # DSEC detection labels: pedestrian, two-wheeler, vehicle.
SEG_CLASSES = 19
SEG_IGNORE = 255


class OperationEnergyMeter:
    """Measure encoder activity and report the same 45nm proxy as training."""

    def __init__(self, model: nn.Module) -> None:
        self.rows: dict[str, dict[str, float]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name: str, module: nn.Module):
        def hook(_module, inputs, output):
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            x = inputs[0].detach()
            if isinstance(module, nn.Linear):
                kernel_ops = module.in_features
            else:
                kh, kw = module.kernel_size
                kernel_ops = kh * kw * (module.in_channels // module.groups)
            dense = float(output[0].numel() * kernel_ops)
            activity = float(x.float().ne(0).float().mean())
            row = self.rows.setdefault(
                name, {"calls": 0.0, "dense": 0.0, "synops": 0.0}
            )
            row["calls"] += 1.0
            row["dense"] += dense
            row["synops"] += dense * activity
        return hook

    def reset(self) -> None:
        self.rows.clear()

    def report(self, batches: int) -> dict[str, float | int]:
        batches = max(1, int(batches))
        dense = sum(row["dense"] for row in self.rows.values()) / batches
        synops = sum(row["synops"] for row in self.rows.values()) / batches
        calls = sum(row["calls"] for row in self.rows.values()) / batches
        ac_pj, mac_pj = 0.9, 4.6
        return {
            "synops_per_sample": float(synops),
            "dense_ops_per_sample": float(dense),
            "activity_fraction": float(synops / max(dense, 1.0)),
            "estimated_energy_pj": float(synops * ac_pj),
            "dense_mac_energy_pj": float(dense * mac_pj),
            "estimated_energy_ratio": float(
                (synops * ac_pj) / max(dense * mac_pj, 1.0)
            ),
            "hook_calls_per_sample": int(calls),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _ids(manifest: Path, split: str) -> set[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    key = "train_ids" if split == "train" else "validation_ids"
    return set(map(str, payload.get(key, [])))


class DSECDownstream(Dataset):
    """DSEC events paired with official detection and segmentation labels."""

    def __init__(self, root: Path, annotations: Path, manifest: Path,
                 split: str, steps: int = 16, size: int = 48) -> None:
        self.root, self.annotations, self.split = Path(root), Path(annotations), split
        # DSEC's official validation annotations are a subset of the train
        # sequences; the representation loader calls that split ``test``.
        # Build the complete train index, then apply the explicit manifest IDs.
        self.events = DSECEvents(root, True, steps=steps, size=size)
        wanted = _ids(manifest, split)
        # Semantic samples must always come from official mask files; the
        # downloaded subset contains only a small intersection with them.
        with zipfile.ZipFile(self.annotations / "train_semantic_segmentation.zip") as z:
            semantic_ids = {
                n[:-4].replace("/19classes/", "/")
                for n in z.namelist() if n.endswith(".png") and "/19classes/" in n
            }
        # Semantic benchmark labels cover only the Zurich training sequences.
        # Hold out the last 10% per sequence, and remove those IDs from head
        # training so downstream labels never leak into validation.
        by_seq: dict[str, list[str]] = {}
        for sid in sorted(semantic_ids):
            by_seq.setdefault(sid.split("/")[1], []).append(sid)
        available_sequences = {row[0] for row in self.events.samples}
        validation_ids: set[str] = set()
        for seq, ids in by_seq.items():
            if seq in available_sequences:
                validation_ids.update(ids[-max(1, len(ids) // 10):])
        if split == "train":
            wanted = (wanted & semantic_ids) - validation_ids
        if split == "validation":
            wanted = validation_ids
        self.samples = []
        self._event_indices: list[int] = []
        self._det_zips: dict[str, zipfile.ZipFile] = {}
        self._seg_zip = None
        # Keep only samples with at least one official downstream annotation.
        for event_index, row in enumerate(self.events.samples):
            seq, frame, timestamp, event_zip, image_zip = row
            # Official archives use the train/<sequence>/<frame> namespace.
            sid = f"train/{seq}/{frame:06d}"
            if sid in wanted:
                self.samples.append(row)
                self._event_indices.append(event_index)
        if not self.samples:
            raise RuntimeError(f"no {split} samples matched the split manifest")

    def __len__(self) -> int:
        return len(self.samples)

    def _det(self, seq: str, frame: int, timestamp: int) -> np.ndarray:
        if seq not in self._det_zips:
            with zipfile.ZipFile(
                self.annotations / "dsec-det_left_object_detections.zip"
            ) as outer:
                nested = zipfile.ZipFile(
                    io.BytesIO(outer.read("train_object_detections.zip"))
                )
            self._det_zips[seq] = nested
        z = self._det_zips[seq]
        path = f"train/{seq}/object_detections/left/tracks.npy"
        if path not in z.namelist():
            return np.zeros((0, 4), np.float32)
        records = np.load(io.BytesIO(z.read(path)), allow_pickle=False)
        # tracks.npy stores asynchronous boxes.  Use the nearest timestamp to
        # the RGB/event frame and return xyxy + class id.
        if len(records) == 0:
            return np.zeros((0, 5), np.float32)
        nearest = np.abs(records["t"].astype(np.int64) - int(timestamp))
        t0 = records["t"][int(nearest.argmin())]
        rows = records[records["t"] == t0]
        out = np.zeros((len(rows), 5), np.float32)
        out[:, 0] = rows["x"]
        out[:, 1] = rows["y"]
        out[:, 2] = rows["x"] + rows["w"]
        out[:, 3] = rows["y"] + rows["h"]
        out[:, 4] = rows["class_id"]
        return out

    def _seg(self, seq: str, frame: int) -> np.ndarray:
        name = f"train/{seq}/19classes/{frame:06d}.png"
        if self._seg_zip is None:
            self._seg_zip = zipfile.ZipFile(
                self.annotations / "train_semantic_segmentation.zip"
            )
        if name not in self._seg_zip.NameToInfo:
            return np.full((48, 48), SEG_IGNORE, np.uint8)
        with Image.open(io.BytesIO(self._seg_zip.read(name))) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        return mask

    def __getitem__(self, index: int):
        seq, frame, timestamp, *_ = self.samples[index]
        # DSECEvents uses the same deterministic sample ordering.
        event, _, _, sid = self.events[self._event_indices[index]]
        return event, self._det(seq, frame, timestamp), self._seg(seq, frame), sid


class DetectionHead(nn.Module):
    def __init__(self, channels: int, classes: int = DET_CLASSES) -> None:
        super().__init__()
        self.tower = DenseTokenDecoder(channels)
        self.cls = nn.Conv2d(64, classes, 1)
        self.box = nn.Conv2d(64, 4, 1)
        nn.init.constant_(self.cls.bias, -4.595)

    def forward(self, x):
        y = self.tower(x)
        return self.cls(y), F.softplus(self.box(y))


class SegmentationHead(nn.Module):
    def __init__(self, channels: int, classes: int = SEG_CLASSES) -> None:
        super().__init__()
        self.decoder = DenseTokenDecoder(channels)
        self.classifier = nn.Conv2d(64, classes, 1)

    def forward(self, x, output_size):
        logits = self.classifier(self.decoder(x))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DenseTokenDecoder(nn.Module):
    """Recover a 48x48 task grid from the frozen 12x12 token lattice."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 96),
            nn.GELU(),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(0.0, rb - lt), axis=-1)
    area_a = np.prod(np.maximum(0.0, a[:, 2:] - a[:, :2]), axis=-1)
    area_b = np.prod(np.maximum(0.0, b[:, 2:] - b[:, :2]), axis=-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _dsec_class(class_id: int) -> int | None:
    """Collapse DSEC's seven labels into the three benchmark superclasses."""
    class_id = int(class_id)
    if class_id == 0:  # pedestrian
        return 0
    if class_id in {1, 5, 6}:  # rider, bicycle, motorcycle
        return 1
    if class_id in {2, 3, 4}:  # car, bus, truck
        return 2
    return None


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order):
        index = int(order[0])
        keep.append(index)
        if len(order) == 1:
            break
        overlap = _box_iou(boxes[index:index + 1], boxes[order[1:]])[0]
        order = order[1:][overlap < threshold]
    return keep


def _average_precision(rows: list[tuple[float, bool]], gt_count: int) -> float:
    """All-point interpolated AP with the standard recall end points."""
    if not rows or gt_count <= 0:
        return 0.0
    ordered = sorted(rows, key=lambda row: row[0], reverse=True)
    tp = np.cumsum([int(hit) for _, hit in ordered], dtype=np.float64)
    fp = np.cumsum([int(not hit) for _, hit in ordered], dtype=np.float64)
    recall = tp / float(gt_count)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[changed] - recall[changed - 1]) * precision[changed]))


@torch.no_grad()
def _validate(encoder, det, seg, loader, device, energy_meter, head_energy_meter) -> dict[str, object]:
    det.eval(); seg.eval()
    energy_meter.reset()
    head_energy_meter.reset()
    confusion = np.zeros((SEG_CLASSES, SEG_CLASSES), np.int64)
    scores: list[list[tuple[float, bool]]] = [[] for _ in range(DET_CLASSES)]
    gt_count = [0] * DET_CLASSES
    for rows in loader:
        events = torch.stack([r[0] for r in rows]).to(device)
        boxes = [r[1] for r in rows]
        masks = np.stack([r[2] for r in rows])
        feat = _features(encoder, events)
        logits, reg = det(feat)
        pred_masks = seg(feat, masks.shape[-2:]).argmax(1).cpu().numpy()
        valid = masks != SEG_IGNORE
        pair = SEG_CLASSES * masks[valid].astype(np.int64) + pred_masks[valid]
        confusion += np.bincount(pair, minlength=SEG_CLASSES * SEG_CLASSES).reshape(SEG_CLASSES, SEG_CLASSES)
        prob = logits.sigmoid().cpu().numpy()
        reg_np = reg.cpu().numpy()
        for n, gt in enumerate(boxes):
            for cls in range(DET_CLASSES):
                truth = np.asarray([
                    row[:4] for row in gt if _dsec_class(int(row[4])) == cls
                ], dtype=np.float32).reshape(-1, 4)
                gt_count[cls] += len(truth)
                used = np.zeros(len(truth), bool)
                flat_scores = prob[n, cls].reshape(-1)
                candidate_count = min(200, len(flat_scores))
                flat_indices = np.argpartition(
                    flat_scores, len(flat_scores) - candidate_count
                )[-candidate_count:]
                candidates = np.column_stack(np.unravel_index(
                    flat_indices, prob[n, cls].shape
                ))
                predicted_boxes = []
                predicted_scores = []
                for iy, ix in candidates:
                    cx = (ix + reg_np[n, 0, iy, ix]) / prob.shape[-1] * 640
                    cy = (iy + reg_np[n, 1, iy, ix]) / prob.shape[-2] * 480
                    ww = reg_np[n, 2, iy, ix] / prob.shape[-1] * 640
                    hh = reg_np[n, 3, iy, ix] / prob.shape[-2] * 480
                    predicted_boxes.append([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2])
                    predicted_scores.append(float(prob[n, cls, iy, ix]))
                predicted_boxes = np.asarray(predicted_boxes, np.float32).reshape(-1, 4)
                predicted_scores = np.asarray(predicted_scores, np.float32)
                for prediction_index in _nms(predicted_boxes, predicted_scores)[:100]:
                    box = predicted_boxes[prediction_index:prediction_index + 1]
                    ious = _box_iou(box, truth)[0] if len(truth) else np.zeros(0)
                    hit = bool(len(ious) and ious.max() >= 0.5)
                    if hit:
                        j = int(ious.argmax())
                        if used[j]:
                            hit = False
                        else:
                            used[j] = True
                    scores[cls].append((float(predicted_scores[prediction_index]), hit))
    union = confusion.sum(1) + confusion.sum(0) - np.diag(confusion)
    ious = np.diag(confusion) / np.maximum(union, 1)
    present = confusion.sum(1) > 0
    ap = []
    for cls in range(DET_CLASSES):
        if gt_count[cls] == 0:
            continue
        ap.append(_average_precision(scores[cls], gt_count[cls]))
    return {"seg_miou": float(np.mean(ious)),
            "seg_miou_present": float(np.mean(ious[present])) if present.any() else 0.0,
            "seg_present_classes": int(present.sum()),
            "det_ap50_proxy": float(np.mean(ap)) if ap else 0.0,
            "det_gt": float(sum(gt_count)), "encoder_energy": energy_meter.report(len(loader)),
            "downstream_head_energy": head_energy_meter.report(len(loader))}


@torch.no_grad()
def _save_visuals(encoder, det, seg, dataset: DSECDownstream,
                  output_dir: Path, device: torch.device, limit: int = 4) -> None:
    """Save qualitative panels without changing the frozen encoder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    det.eval(); seg.eval()
    for index in range(min(limit, len(dataset))):
        events, gt_boxes, gt_mask, sample_id = dataset[index]
        feat = _features(encoder, events.unsqueeze(0).to(device))
        logits, reg = det(feat)
        pred_mask = seg(feat, gt_mask.shape).argmax(1)[0].cpu().numpy().astype(np.uint8)
        prob = logits.sigmoid()[0].cpu().numpy()
        reg_np = reg[0].cpu().numpy()
        pred_boxes: list[tuple[float, float, float, float, int, float]] = []
        for cls in range(DET_CLASSES):
            ys, xs = np.where(prob[cls] > 0.25)
            order = np.argsort(prob[cls, ys, xs])[::-1][:20]
            for oi in order:
                y, x = int(ys[oi]), int(xs[oi])
                cx = (x + reg_np[0, y, x]) / prob.shape[-1] * 640
                cy = (y + reg_np[1, y, x]) / prob.shape[-2] * 480
                ww = reg_np[2, y, x] / prob.shape[-1] * 640
                hh = reg_np[3, y, x] / prob.shape[-2] * 480
                pred_boxes.append((cx - ww / 2, cy - hh / 2, cx + ww / 2,
                                   cy + hh / 2, cls, float(prob[cls, y, x])))
        event_img = events.sum(0).sum(0).clamp(0, 1).mul(255).byte().cpu().numpy()
        event_img = Image.fromarray(event_img).resize((640, 440), Image.Resampling.NEAREST).convert("RGB")

        def panel_boxes(boxes, title, predicted=False):
            image = event_img.copy(); draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 639, 27), fill=(0, 0, 0)); draw.text((8, 7), title, fill=(255, 255, 255))
            for row in boxes:
                x1, y1, x2, y2 = map(float, row[:4])
                color = (255, 80, 80) if not predicted else (80, 255, 120)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                if predicted:
                    draw.text((x1, max(28, y1)), f"c{int(row[4])}:{float(row[5]):.2f}", fill=color)
            return image

        def panel_mask(mask, title):
            palette = np.zeros((256, 3), np.uint8)
            palette[:19] = np.array([(0, 0, 0), (128, 64, 128), (70, 70, 70), (220, 20, 60),
                                     (0, 0, 142), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                                     (107, 142, 35), (152, 251, 152), (70, 130, 180), (255, 0, 0),
                                     (0, 60, 100), (0, 0, 70), (0, 80, 100), (0, 0, 230),
                                     (119, 11, 32), (244, 35, 232), (190, 153, 153)])
            rgb = Image.fromarray(palette[mask], "RGB").resize((640, 440), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(rgb); draw.rectangle((0, 0, 639, 27), fill=(0, 0, 0)); draw.text((8, 7), title, fill=(255, 255, 255))
            return rgb

        panels = [panel_boxes(gt_boxes, "event + GT boxes"), panel_boxes(pred_boxes, "event + predicted boxes", True),
                  panel_mask(gt_mask, "GT semantic mask"), panel_mask(pred_mask, "predicted semantic mask")]
        canvas = Image.new("RGB", (1280, 880), (20, 20, 20))
        for pos, image in enumerate(panels):
            canvas.paste(image, ((pos % 2) * 640, (pos // 2) * 440))
        safe = sample_id.replace("/", "_")
        canvas.save(output_dir / f"{index:02d}_{safe}.png")


def _encoder(checkpoint: Path, device: torch.device) -> PureSpikeFormer:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_args = payload["args"]
    args = argparse.Namespace(**raw_args) if isinstance(raw_args, dict) else raw_args
    model = PureSpikeFormer(in_channels=2, dim=args.dim, depth=args.depth, heads=args.heads,
                            patch_size=args.patch_size, threshold=args.threshold,
                            use_cls_token=args.use_cls_token, norm="bntt", image_size=args.size,
                            temporal_readout="learned", local_stem=not args.pyramid_stem,
                            pyramid_stem=args.pyramid_stem, use_positional_bias=True,
                            attention_mode="normalized", signed_readout=args.signed_readout,
                            multidepth_readout=args.multidepth_readout,
                            multidepth_readout_layers=args.multidepth_readout_layers,
                            membrane_readout=args.membrane_readout, block_spiking=args.block_spiking,
                            hybrid_attention=args.hybrid_attention,
                            hybrid_attention_suffix=args.hybrid_attention_suffix,
                            qkv_temporal_mode=args.qkv_temporal_mode, continuous=args.continuous_relaxation,
                            ternary_threshold=args.ternary_threshold, temporal_steps=args.steps,
                            population_bins=args.population_bins,
                            event_prompt=bool(getattr(args, "event_prompt", False)),
                            prompt_strength=float(getattr(args, "prompt_strength", 0.7)),
                            local_structure_mixer=bool(getattr(args, "local_structure_mixer", False)),
                            local_mixer_dilations=tuple(getattr(args, "local_mixer_dilations", [1, 2]))).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def _features(encoder, events: torch.Tensor) -> torch.Tensor:
    out = encoder(events, return_tokens=True)
    tokens = out["tokens"].float().mean(dim=1)
    side = math.isqrt(tokens.shape[1])
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)


def _det_targets(boxes: list[np.ndarray], h: int, w: int, device):
    cls = torch.zeros((len(boxes), DET_CLASSES, h, w), device=device)
    reg = torch.zeros((len(boxes), 4, h, w), device=device)
    positive = torch.zeros((len(boxes), h, w), dtype=torch.bool, device=device)
    for n, rows in enumerate(boxes):
        for x1, y1, x2, y2, label in rows:
            target_class = _dsec_class(int(label))
            if target_class is None:
                continue
            cx, cy = ((x1 + x2) * .5 / 640 * w), ((y1 + y2) * .5 / 480 * h)
            ix, iy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
            width = max(1.0, (x2 - x1) / 640.0 * w)
            height = max(1.0, (y2 - y1) / 480.0 * h)
            radius_x = max(1, int(round(0.3 * width)))
            radius_y = max(1, int(round(0.3 * height)))
            x0, x1i = max(0, ix - radius_x), min(w, ix + radius_x + 1)
            y0, y1i = max(0, iy - radius_y), min(h, iy + radius_y + 1)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(y0, y1i, device=device),
                torch.arange(x0, x1i, device=device), indexing="ij",
            )
            gaussian = torch.exp(
                -0.5 * (
                    ((grid_x - cx) / max(radius_x / 3.0, 0.5)).square()
                    + ((grid_y - cy) / max(radius_y / 3.0, 0.5)).square()
                )
            )
            cls[n, target_class, y0:y1i, x0:x1i] = torch.maximum(
                cls[n, target_class, y0:y1i, x0:x1i], gaussian
            )
            cls[n, target_class, iy, ix] = 1.0
            positive[n, iy, ix] = True
            reg[n, :, iy, ix] = torch.tensor(
                [cx - ix, cy - iy, width, height], device=device
            )
    return cls, reg, positive


def _center_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    positive = targets.eq(1.0)
    negative = targets.lt(1.0)
    negative_weight = (1.0 - targets).pow(4)
    positive_loss = -(1.0 - probabilities).pow(2) * probabilities.log() * positive
    negative_loss = (
        -probabilities.pow(2) * (1.0 - probabilities).log()
        * negative_weight * negative
    )
    count = positive.sum().clamp_min(1)
    return (positive_loss.sum() + negative_loss.sum()) / count


def _ohem_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, hard_fraction: float = 0.25,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits, targets, ignore_index=SEG_IGNORE, reduction="none"
    )
    valid_losses = losses[targets.ne(SEG_IGNORE)]
    if not len(valid_losses):
        return losses.sum() * 0.0
    count = max(1, int(math.ceil(float(hard_fraction) * len(valid_losses))))
    return valid_losses.topk(count, sorted=False).values.mean()


def _downstream_training_regime(history: list[dict[str, object]]) -> dict[str, object]:
    evaluated = [row for row in history if row.get("seg_miou") is not None]
    if len(evaluated) < 3:
        return {
            "state": "insufficient_evidence",
            "evaluations": len(evaluated),
            "required_evaluations": 3,
            "recommendation": "continue_to_next_validation",
        }
    window = evaluated[-3:]
    segmentation = [float(row["seg_miou"]) for row in window]
    detection = [float(row["det_ap50_proxy"]) for row in window]
    losses = [float(row["loss"]) for row in window]
    seg_delta = segmentation[-1] - segmentation[0]
    det_delta = detection[-1] - detection[0]
    loss_reduction = (losses[0] - losses[-1]) / max(abs(losses[0]), 1.0e-8)
    if seg_delta >= 0.01 or det_delta >= 0.01:
        state, recommendation = "undertrained_improving", "continue_training"
    elif seg_delta <= -0.005 or det_delta <= -0.005:
        state = "downstream_overfit_or_optimization_conflict"
        recommendation = "stop_and_rebalance_downstream_objectives"
    elif abs(seg_delta) < 0.005 and abs(det_delta) < 0.005 and loss_reduction < 0.02:
        state = "converged_plateau"
        recommendation = "stop_and_change_head_or_representation"
    elif loss_reduction >= 0.02:
        state = "objective_mismatch"
        recommendation = "do_not_add_epochs_without_metric_or_loss_fix"
    else:
        state = "ambiguous_slow_progress"
        recommendation = "continue_one_validation_interval_only"
    return {
        "state": state,
        "recommendation": recommendation,
        "evaluation_epochs": [int(row["epoch"]) for row in window],
        "seg_miou_delta": seg_delta,
        "det_ap50_proxy_delta": det_delta,
        "relative_loss_reduction": loss_reduction,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="run the downstream probe without CUDA")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--visual-limit", type=int, default=4)
    p.add_argument("--ohem-fraction", type=float, default=0.25)
    a = p.parse_args()
    if not 0.0 < a.ohem_fraction <= 1.0:
        raise ValueError("--ohem-fraction must be in (0, 1]")
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else f"cuda:{a.gpu}")
    train = DSECDownstream(a.data_root, a.annotations, a.split_manifest, "train")
    valid = DSECDownstream(a.data_root, a.annotations, a.split_manifest, "validation")
    train_ids = {f"train/{row[0]}/{row[1]:06d}" for row in train.samples}
    validation_ids = {f"train/{row[0]}/{row[1]:06d}" for row in valid.samples}
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"downstream train/validation leakage: {len(overlap)} IDs")
    if a.smoke:
        train.samples, valid.samples = train.samples[:2], valid.samples[:2]
    loader = DataLoader(train, a.batch_size, shuffle=True, num_workers=a.workers,
                        collate_fn=lambda rows: rows, pin_memory=True,
                        persistent_workers=a.workers > 0)
    validation_loader = DataLoader(valid, a.batch_size, shuffle=False, num_workers=a.workers,
                                   collate_fn=lambda rows: rows, pin_memory=True,
                                   persistent_workers=a.workers > 0)
    encoder = _encoder(a.checkpoint, device)
    det, seg = DetectionHead(encoder.dim).to(device), SegmentationHead(encoder.dim).to(device)
    energy_meter = OperationEnergyMeter(encoder)
    head_energy_meter = OperationEnergyMeter(nn.ModuleList([det, seg]))
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    downstream_parameters = sum(parameter.numel() for parameter in det.parameters()) + sum(
        parameter.numel() for parameter in seg.parameters()
    )
    optimizer = torch.optim.AdamW(list(det.parameters()) + list(seg.parameters()), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=a.epochs, eta_min=1.0e-5
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_seg_miou = -1.0
    best_seg_epoch = 0
    best_det_ap50 = -1.0
    best_det_epoch = 0
    for epoch in range(1, a.epochs + 1):
        det.train(); seg.train(); total = det_total = seg_total = 0.
        for rows in loader:
            events = torch.stack([r[0] for r in rows]).to(device)
            boxes = [r[1] for r in rows]
            masks = torch.from_numpy(np.stack([r[2] for r in rows])).long().to(device)
            feat = _features(encoder, events)
            logits, reg = det(feat)
            cls_t, reg_t, positive = _det_targets(boxes, logits.shape[-2], logits.shape[-1], device)
            det_loss = _center_focal_loss(logits, cls_t)
            if positive.any():
                regression_mask = positive.unsqueeze(1).expand_as(reg)
                det_loss = det_loss + F.smooth_l1_loss(
                    reg[regression_mask], reg_t[regression_mask]
                )
            seg_logits = seg(feat, masks.shape[-2:])
            seg_loss = _ohem_cross_entropy(
                seg_logits, masks, hard_fraction=a.ohem_fraction
            )
            loss = det_loss + seg_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())
            det_total += float(det_loss.detach())
            seg_total += float(seg_loss.detach())
        row = {
            "epoch": epoch,
            "loss": total / max(1, len(loader)),
            "detection_loss": det_total / max(1, len(loader)),
            "segmentation_loss": seg_total / max(1, len(loader)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_samples": len(train), "validation_samples": len(valid),
        }
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            row.update(_validate(
                encoder, det, seg, validation_loader, device,
                energy_meter, head_energy_meter,
            ))
            row.update({"encoder_parameters": encoder_parameters,
                        "downstream_parameters": downstream_parameters})
            row["training_regime"] = _downstream_training_regime([*history, row])
            if float(row["seg_miou"]) > best_seg_miou:
                best_seg_miou, best_seg_epoch = float(row["seg_miou"]), epoch
                torch.save({"segmentation": seg.state_dict(), "epoch": epoch,
                            "metrics": row, "checkpoint": str(a.checkpoint)},
                           a.output_dir / "best_segmentation_head.pt")
            if float(row["det_ap50_proxy"]) > best_det_ap50:
                best_det_ap50, best_det_epoch = float(row["det_ap50_proxy"]), epoch
                torch.save({"detection": det.state_dict(), "epoch": epoch,
                            "metrics": row, "checkpoint": str(a.checkpoint)},
                           a.output_dir / "best_detection_head.pt")
            row["best_segmentation_epoch"] = best_seg_epoch
            row["best_detection_epoch"] = best_det_epoch
        history.append(row)
        print(json.dumps(history[-1]), flush=True)
        (a.output_dir / "downstream_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            torch.save({"detection": det.state_dict(), "segmentation": seg.state_dict(),
                        "checkpoint": str(a.checkpoint), "train_samples": len(train),
                        "validation_samples": len(valid), "epoch": epoch,
                        "metrics": row}, a.output_dir / "downstream_heads.pt")
        scheduler.step()
    final_row = history[-1]
    torch.save({"detection": det.state_dict(), "segmentation": seg.state_dict(),
                "checkpoint": str(a.checkpoint), "train_samples": len(train),
                "validation_samples": len(valid), "epoch": final_row["epoch"],
                "metrics": final_row}, a.output_dir / "downstream_heads.pt")
    selected_det = torch.load(
        a.output_dir / "best_detection_head.pt", map_location=device, weights_only=False
    )
    selected_seg = torch.load(
        a.output_dir / "best_segmentation_head.pt", map_location=device, weights_only=False
    )
    det.load_state_dict(selected_det["detection"])
    seg.load_state_dict(selected_seg["segmentation"])
    torch.save({
        "detection": det.state_dict(), "segmentation": seg.state_dict(),
        "detection_epoch": selected_det["epoch"],
        "segmentation_epoch": selected_seg["epoch"],
        "checkpoint": str(a.checkpoint), "protocol": "validation_selected_heads",
    }, a.output_dir / "selected_downstream_heads.pt")
    if a.visualize:
        _save_visuals(encoder, det, seg, valid, a.output_dir / "visuals", device, a.visual_limit)
    (a.output_dir / "downstream_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    energy_meter.close()
    head_energy_meter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
