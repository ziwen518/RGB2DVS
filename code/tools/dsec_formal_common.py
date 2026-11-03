"""Shared data and frozen-encoder utilities for formal DSEC downstream tasks."""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core"
sys.path.insert(0, str(CORE if CORE.exists() else HERE))
from data import DSECEvents  # noqa: E402
from model import PureSpikeFormer  # noqa: E402


DET_CLASSES = 3
SEG_CLASSES = 19
SEG_IGNORE = 255
EVENT_HEIGHT = 48
EVENT_WIDTH = 48
DSEC_HEIGHT = 480
DSEC_WIDTH = 640


def manifest_ids(path: Path, split: str) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    key = "train_ids" if split == "train" else "validation_ids"
    return set(map(str, payload.get(key, [])))


def dsec_class(class_id: int) -> int | None:
    """Map seven DSEC labels to pedestrian/two-wheeler/vehicle indices."""
    class_id = int(class_id)
    if class_id == 0:
        return 0
    if class_id in {1, 5, 6}:
        return 1
    if class_id in {2, 3, 4}:
        return 2
    return None


class _DSECBase(Dataset):
    def __init__(self, root: Path, split: str, steps: int = 16, size: int = 48) -> None:
        self.root = Path(root)
        self.split = split
        self.events = DSECEvents(self.root, True, steps=steps, size=size)
        self.samples: list[tuple] = []
        self.event_indices: list[int] = []

    def _append(self, wanted: set[str]) -> None:
        for event_index, row in enumerate(self.events.samples):
            seq, frame, *_ = row
            sid = f"train/{seq}/{frame:06d}"
            if sid in wanted:
                self.samples.append(row)
                self.event_indices.append(event_index)
        if not self.samples:
            raise RuntimeError(f"no DSEC {self.split} samples matched")

    def sample_ids(self) -> set[str]:
        return {
            f"train/{row[0]}/{row[1]:06d}"
            for row in self.samples
        }


class DSECDetectionDataset(_DSECBase):
    """All event samples in the fixed manifest, paired with DSEC tracks."""

    def __init__(
        self, root: Path, annotations: Path, manifest: Path,
        split: str, steps: int = 16, size: int = 48,
    ) -> None:
        super().__init__(root, split, steps=steps, size=size)
        self.annotations = Path(annotations)
        self._nested_detection: zipfile.ZipFile | None = None
        self._track_cache: dict[str, np.ndarray] = {}
        self._append(manifest_ids(manifest, split))

    def _archive(self) -> zipfile.ZipFile:
        if self._nested_detection is None:
            with zipfile.ZipFile(
                self.annotations / "dsec-det_left_object_detections.zip"
            ) as outer:
                payload = outer.read("train_object_detections.zip")
            self._nested_detection = zipfile.ZipFile(io.BytesIO(payload))
        return self._nested_detection

    def _tracks(self, seq: str) -> np.ndarray:
        if seq not in self._track_cache:
            archive = self._archive()
            name = f"train/{seq}/object_detections/left/tracks.npy"
            self._track_cache[seq] = (
                np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
                if name in archive.NameToInfo else np.zeros(0)
            )
        return self._track_cache[seq]

    def _target(self, seq: str, timestamp: int) -> dict[str, torch.Tensor]:
        records = self._tracks(seq)
        rows: list[list[float]] = []
        labels: list[int] = []
        if len(records):
            nearest = np.abs(records["t"].astype(np.int64) - int(timestamp))
            selected_time = records["t"][int(nearest.argmin())]
            for record in records[records["t"] == selected_time]:
                label = dsec_class(int(record["class_id"]))
                if label is None:
                    continue
                x1 = float(record["x"]) / DSEC_WIDTH * EVENT_WIDTH
                y1 = float(record["y"]) / DSEC_HEIGHT * EVENT_HEIGHT
                x2 = float(record["x"] + record["w"]) / DSEC_WIDTH * EVENT_WIDTH
                y2 = float(record["y"] + record["h"]) / DSEC_HEIGHT * EVENT_HEIGHT
                x1, x2 = np.clip([x1, x2], 0.0, EVENT_WIDTH)
                y1, y2 = np.clip([y1, y2], 0.0, EVENT_HEIGHT)
                if x2 > x1 and y2 > y1:
                    rows.append([x1, y1, x2, y2])
                    labels.append(label + 1)  # Faster R-CNN reserves zero for background.
        return {
            "boxes": torch.tensor(rows, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        seq, _frame, timestamp, *_ = self.samples[index]
        events, *_ = self.events[self.event_indices[index]]
        target = self._target(seq, timestamp)
        target["image_id"] = torch.tensor([index], dtype=torch.int64)
        # Torchvision's transform accepts CxHxW images. Preserve time/polarity
        # by flattening T and C; the frozen backbone restores [T, 2].
        image = events.reshape(-1, events.shape[-2], events.shape[-1])
        sid = f"train/{seq}/{self.samples[index][1]:06d}"
        return image, target, sid


class DSECSegmentationDataset(_DSECBase):
    """Deterministic 90/10 per-sequence split over available semantic masks."""

    def __init__(
        self, root: Path, annotations: Path, split: str,
        steps: int = 16, size: int = 48,
    ) -> None:
        super().__init__(root, split, steps=steps, size=size)
        self.annotations = Path(annotations)
        self._seg_zip: zipfile.ZipFile | None = None
        with zipfile.ZipFile(
            self.annotations / "train_semantic_segmentation.zip"
        ) as archive:
            semantic_ids = {
                name[:-4].replace("/19classes/", "/")
                for name in archive.namelist()
                if name.endswith(".png") and "/19classes/" in name
            }
        available = {
            f"train/{row[0]}/{row[1]:06d}"
            for row in self.events.samples
        }
        by_sequence: dict[str, list[str]] = {}
        for sid in sorted(semantic_ids & available):
            by_sequence.setdefault(sid.split("/")[1], []).append(sid)
        train_ids: set[str] = set()
        validation_ids: set[str] = set()
        for ids in by_sequence.values():
            holdout = max(1, len(ids) // 10)
            train_ids.update(ids[:-holdout])
            validation_ids.update(ids[-holdout:])
        if train_ids & validation_ids:
            raise RuntimeError("semantic train/validation overlap")
        self._append(train_ids if split == "train" else validation_ids)

    def _mask(self, seq: str, frame: int) -> torch.Tensor:
        if self._seg_zip is None:
            self._seg_zip = zipfile.ZipFile(
                self.annotations / "train_semantic_segmentation.zip"
            )
        name = f"train/{seq}/19classes/{frame:06d}.png"
        with Image.open(io.BytesIO(self._seg_zip.read(name))) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        return torch.from_numpy(mask).long()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        seq, frame, *_ = self.samples[index]
        events, *_ = self.events[self.event_indices[index]]
        sid = f"train/{seq}/{frame:06d}"
        return events, self._mask(seq, frame), sid


class OperationEnergyMeter:
    """45 nm AC/MAC proxy using runtime activation sparsity."""

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
            source = inputs[0].detach()
            if isinstance(module, nn.Linear):
                kernel_ops = module.in_features
            else:
                kh, kw = module.kernel_size
                kernel_ops = kh * kw * (module.in_channels // module.groups)
            dense = float(output[0].numel() * kernel_ops)
            activity = float(source.float().ne(0).float().mean())
            row = self.rows.setdefault(name, {"calls": 0.0, "dense": 0.0, "synops": 0.0})
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
        return {
            "synops_per_sample": synops,
            "dense_ops_per_sample": dense,
            "activity_fraction": synops / max(dense, 1.0),
            "estimated_energy_pj": synops * 0.9,
            "estimated_energy_mj": synops * 0.9 / 1.0e9,
            "dense_mac_energy_pj": dense * 4.6,
            "estimated_energy_ratio": (synops * 0.9) / max(dense * 4.6, 1.0),
            "hook_calls_per_sample": int(calls),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_encoder(checkpoint: Path, device: torch.device) -> PureSpikeFormer:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = payload["args"]
    cfg = vars(raw) if isinstance(raw, argparse.Namespace) else dict(raw)
    model = PureSpikeFormer(
        in_channels=2, dim=int(cfg["dim"]), depth=int(cfg["depth"]),
        heads=int(cfg["heads"]), patch_size=int(cfg["patch_size"]),
        threshold=float(cfg["threshold"]),
        use_cls_token=bool(cfg.get("use_cls_token", False)), norm="bntt",
        image_size=int(cfg["size"]), temporal_readout="learned",
        local_stem=not bool(cfg.get("pyramid_stem", False)),
        pyramid_stem=bool(cfg.get("pyramid_stem", False)),
        use_positional_bias=True, attention_mode="normalized",
        signed_readout=bool(cfg.get("signed_readout", True)),
        multidepth_readout=bool(cfg.get("multidepth_readout", False)),
        multidepth_readout_layers=int(cfg.get("multidepth_readout_layers", 3)),
        membrane_readout=bool(cfg.get("membrane_readout", False)),
        block_spiking=cfg.get("block_spiking", "full_respike"),
        hybrid_attention=bool(cfg.get("hybrid_attention", False)),
        hybrid_attention_suffix=int(cfg.get("hybrid_attention_suffix", 0)),
        qkv_temporal_mode=cfg.get("qkv_temporal_mode", "standard"),
        continuous=bool(cfg.get("continuous_relaxation", False)),
        ternary_threshold=float(cfg.get("ternary_threshold", 0.5)),
        temporal_steps=int(cfg["steps"]),
        population_bins=int(cfg.get("population_bins", 1)),
        event_prompt=bool(cfg.get("event_prompt", False)),
        prompt_strength=float(cfg.get("prompt_strength", 0.7)),
        local_structure_mixer=bool(cfg.get("local_structure_mixer", False)),
        local_mixer_dilations=tuple(cfg.get("local_mixer_dilations", [1, 2])),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.formal_checkpoint_args = cfg
    return model


def frozen_token_grid(
    encoder: PureSpikeFormer, events: torch.Tensor,
) -> tuple[torch.Tensor, float, bool]:
    encoder.eval()
    with torch.no_grad():
        output = encoder(events, return_tokens=True)
    tokens = output["tokens"].float()
    binary = bool(torch.all((tokens == 0) | (tokens == 1)))
    tokens = tokens.mean(dim=1)
    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise RuntimeError(f"non-square Student token lattice: {tokens.shape[1]}")
    feature = tokens.transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], side, side
    )
    return feature.detach(), float(output["spike_rate"]), binary


def assert_disjoint(train: _DSECBase, validation: _DSECBase) -> None:
    overlap = train.sample_ids() & validation.sample_ids()
    if overlap:
        raise RuntimeError(f"DSEC downstream split leakage: {len(overlap)} IDs")

