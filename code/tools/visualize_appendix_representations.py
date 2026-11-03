"""Generate appendix PCA and spike-attention affinity figures."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cifar10rcls", "ncaltech101"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=40)
    parser.add_argument("--heatmap-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def build_model(model_module, checkpoint: dict, device: torch.device):
    c = checkpoint["args"]
    model = model_module.PureSpikeFormer(
        in_channels=2, dim=int(c["dim"]), depth=int(c["depth"]), heads=int(c["heads"]),
        patch_size=int(c["patch_size"]), threshold=float(c["threshold"]),
        use_cls_token=bool(c.get("use_cls_token", False)), norm="bntt", image_size=int(c["size"]),
        temporal_readout="learned", local_stem=not bool(c.get("pyramid_stem", False)),
        pyramid_stem=bool(c.get("pyramid_stem", False)), use_positional_bias=True,
        attention_mode="normalized", signed_readout=bool(c.get("signed_readout", True)),
        multidepth_readout=bool(c.get("multidepth_readout", False)),
        multidepth_readout_layers=int(c.get("multidepth_readout_layers", 3)),
        block_spiking=c.get("block_spiking", "full_respike"),
        hybrid_attention=bool(c.get("hybrid_attention", False)),
        hybrid_attention_suffix=int(c.get("hybrid_attention_suffix", 0)),
        qkv_temporal_mode=c.get("qkv_temporal_mode", "standard"),
        continuous=bool(c.get("continuous_relaxation", False)),
        ternary_threshold=float(c.get("ternary_threshold", 0.5)), temporal_steps=int(c["steps"]),
        population_bins=int(c.get("population_bins", 1)), event_prompt=bool(c.get("event_prompt", False)),
        prompt_strength=float(c.get("prompt_strength", 0.7)),
        local_structure_mixer=bool(c.get("local_structure_mixer", False)),
        local_mixer_dilations=tuple(c.get("local_mixer_dilations", [1, 2])),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def sample_id(dataset: Dataset, index: int) -> str:
    path = dataset.samples[index][0]
    return str(path.relative_to(dataset.root)).replace("\\", "/")


CIFAR10_CLASS_NAMES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}


def class_name(dataset: Dataset, index: int, dataset_name: str) -> str:
    label = int(dataset[index][1])
    if dataset_name == "CIFAR10-RCLS":
        return CIFAR10_CLASS_NAMES.get(label, str(label))
    identifier = sample_id(dataset, index)
    parts = identifier.split("/")
    return parts[-2] if len(parts) >= 2 else str(label)


def build_dataset(args: argparse.Namespace, data_module, cfg: dict):
    if args.dataset == "cifar10rcls":
        dataset = data_module.make_event_dataset(
            "motion", args.data_root, False, steps=int(cfg["steps"]), size=int(cfg["size"]),
            frame_mode=cfg.get("frame_mode", "count_global"), preload_frames=True,
        )
        split = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
        wanted = split.get("test_ids", [])
    else:
        dataset = data_module.make_event_dataset(
            "ncaltech101", args.data_root, True, steps=int(cfg["steps"]),
            size=int(cfg["size"]), use_all=True,
        )
        split = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
        wanted = split.get("test_ids", split.get("validation_ids", []))
    lookup = {sample_id(dataset, index): index for index in range(len(dataset))}
    indices = [lookup[str(key).replace("\\", "/")] for key in wanted if str(key).replace("\\", "/") in lookup]
    if not indices:
        indices = list(range(len(dataset)))
    return dataset, indices


def choose_balanced(dataset: Dataset, indices: list[int], max_classes: int, per_class: int, seed: int):
    groups: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        groups[int(dataset[index][1])].append(index)
    classes = sorted(groups, key=lambda value: (-len(groups[value]), value))[:max_classes]
    classes.sort()
    rng = random.Random(seed)
    chosen = []
    for label in classes:
        candidates = groups[label][:]
        rng.shuffle(candidates)
        chosen.extend(candidates[:per_class])
    return chosen, classes


@torch.inference_mode()
def extract_layers(model, dataset: Dataset, indices: list[int], batch_size: int, device: torch.device):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    layers, core, signed, labels = [], [], [], []
    for events, target, *_ in loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(events.to(device, non_blocking=True), return_layers=True)
        layers.append(output["layer_embeddings"].float().cpu())
        core.append(output["core_embedding"].float().cpu())
        signed.append(output["embedding"].float().cpu())
        labels.append(torch.as_tensor(target))
    stacked = torch.cat(layers)
    block_two = min(2, stacked.shape[1] - 1)
    features = {
        "Stem": stacked[:, 0],
        "Block 2": stacked[:, block_two],
        "Core": torch.cat(core),
        "Signed readout": torch.cat(signed),
    }
    return features, torch.cat(labels)


def plot_layerwise_tsne(features: dict[str, torch.Tensor], labels: torch.Tensor, output_dir: Path, dataset_name: str):
    names = list(features)
    normalized = [F.normalize(features[name].float(), dim=-1).numpy() for name in names]
    labels_np = labels.numpy()
    classes = sorted(np.unique(labels_np).tolist())
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(names), figsize=(13.2, 3.35), dpi=220)
    for panel, name, values in zip(axes, names, normalized):
        perplexity = min(30.0, max(5.0, (len(values) - 1) / 3.0))
        tsne_args = dict(
            n_components=2, perplexity=perplexity, init="pca",
            learning_rate="auto", random_state=42,
        )
        try:
            estimator = TSNE(max_iter=1000, **tsne_args)
        except TypeError:
            estimator = TSNE(n_iter=1000, **tsne_args)
        panel_points = estimator.fit_transform(values)
        for color_index, label in enumerate(classes):
            mask = labels_np == label
            panel.scatter(panel_points[mask, 0], panel_points[mask, 1], s=9, alpha=0.72,
                          color=colors(color_index % 10), linewidths=0)
        panel.set_title(name, fontsize=9.5)
        panel.set_xticks([]); panel.set_yticks([])
        for spine in panel.spines.values(): spine.set_color("#cccccc")
    fig.suptitle(f"{dataset_name}: layer-wise representation t-SNE", fontsize=11)
    fig.text(0.5, 0.01, "t-SNE is fit independently at each stage; colors denote class labels.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_dir / "layerwise_tsne.png", bbox_inches="tight")
    fig.savefig(output_dir / "layerwise_tsne.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"method": "independent_tsne", "random_state": 42}


def output_tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


class QKCapture:
    def __init__(self, model):
        self.values = {index: {"q": [], "k": []} for index in range(len(model.blocks))}
        self.handles = []
        for index, block in enumerate(model.blocks):
            self.handles.append(block.attn.q_lif.register_forward_hook(self._hook(index, "q")))
            self.handles.append(block.attn.k_lif.register_forward_hook(self._hook(index, "k")))

    def _hook(self, index: int, key: str):
        def capture(_module, _inputs, output):
            self.values[index][key].append(output_tensor(output).detach().float().cpu())
        return capture

    def clear(self):
        for block in self.values.values():
            block["q"].clear(); block["k"].clear()

    def close(self):
        for handle in self.handles: handle.remove()


def token_importance(q_values: list[torch.Tensor], k_values: list[torch.Tensor], heads: int):
    steps = min(len(q_values), len(k_values))
    q = torch.stack(q_values[:steps])
    k = torch.stack(k_values[:steps])
    time, batch, tokens, channels = q.shape
    head_dim = channels // heads
    q = q.reshape(time, batch, tokens, heads, head_dim)
    k = k.reshape(time, batch, tokens, heads, head_dim)
    mean_query = q.mean(dim=2, keepdim=True)
    score = (k * mean_query).sum(dim=-1).mean(dim=(0, 3))[0]
    side = math.isqrt(tokens)
    if side * side != tokens:
        side = math.isqrt(tokens - 1)
        score = score[-side * side :]
    score = score.reshape(side, side)
    score = (score - score.min()) / (score.max() - score.min()).clamp_min(1.0e-8)
    return score


def event_rgb(events: torch.Tensor):
    on = events[:, 0].sum(0).float()
    off = events[:, 1].sum(0).float()
    on = on / on.max().clamp_min(1.0)
    off = off / off.max().clamp_min(1.0)
    return torch.stack([on, 0.25 * (on + off), off], dim=-1).numpy()


def select_visual_indices(dataset: Dataset, indices: list[int], classes: list[int], count: int):
    selected = []
    for label in classes:
        candidates = [index for index in indices if int(dataset[index][1]) == label]
        if candidates:
            ranked = sorted(
                candidates[: min(30, len(candidates))],
                key=lambda index: float(dataset[index][0].sum()), reverse=True,
            )
            selected.append(ranked[0])
        if len(selected) == count:
            break
    return selected


class SemanticSpikeCapture:
    def __init__(self, model):
        self.values: list[torch.Tensor] = []
        self.handle = model.semantic_lif.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.values.append(output_tensor(output).detach().float().cpu())

    def clear(self):
        self.values.clear()

    def close(self):
        self.handle.remove()


def pca_rgb_projection(features: list[torch.Tensor]):
    counts = [value.shape[0] for value in features]
    matrix = torch.cat(features).float().numpy()
    pca = PCA(n_components=3, random_state=42)
    projected = pca.fit_transform(matrix).astype(np.float32)
    for channel in range(3):
        low, high = np.percentile(projected[:, channel], [1.0, 99.0])
        projected[:, channel] = np.clip(
            (projected[:, channel] - low) / max(float(high - low), 1.0e-8), 0.0, 1.0,
        )
    split = []
    offset = 0
    for count in counts:
        split.append(projected[offset : offset + count])
        offset += count
    return split, pca.explained_variance_ratio_.tolist()


@torch.inference_mode()
def plot_patch_pca_rgb(model, dataset: Dataset, selected: list[int], device: torch.device,
                       output_dir: Path, dataset_name: str, rgb_root: Path | None = None):
    capture = SemanticSpikeCapture(model)
    stage_names = ["Block 1", "Block 2", "Block 4", "Signed readout"]
    per_stage: list[list[torch.Tensor]] = [[] for _ in stage_names]
    event_images = []
    rgb_images = []
    labels = []
    sample_ids = []
    sides = []
    for index in selected:
        events, label, *_ = dataset[index]
        capture.clear()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                events.unsqueeze(0).to(device), return_token_layers=len(model.blocks),
            )
        pyramid = output["token_pyramid"][0].float().cpu().mean(dim=1)
        block_indices = [0, min(1, len(model.blocks) - 1), len(model.blocks) - 1]
        for destination, block_index in enumerate(block_indices):
            per_stage[destination].append(pyramid[block_index])
        semantic = torch.stack(capture.values)
        positive, negative = semantic.chunk(2, dim=-1)
        per_stage[-1].append((positive - negative).mean(dim=0)[0])
        event_images.append(event_rgb(events))
        identifier = sample_id(dataset, index)
        if rgb_root is not None:
            rgb_path = rgb_root / Path(identifier).with_suffix(".png")
            if not rgb_path.is_file():
                raise FileNotFoundError(f"missing paired RGB image: {rgb_path}")
            rgb_images.append(np.asarray(Image.open(rgb_path).convert("RGB")))
        labels.append(class_name(dataset, index, dataset_name))
        sample_ids.append(identifier)
        sides.append(math.isqrt(pyramid.shape[-2]))
    capture.close()

    projected_stages = []
    explained = {}
    for name, values in zip(stage_names, per_stage):
        projected, variance = pca_rgb_projection(values)
        projected_stages.append(projected)
        explained[name] = variance

    has_rgb = bool(rgb_images)
    columns = 1 + int(has_rgb) + len(stage_names)
    fig, axes = plt.subplots(len(selected), columns, figsize=(2.25 * columns, 2.05 * len(selected)), dpi=220)
    axes = np.atleast_2d(axes)
    for row, (base, side, label) in enumerate(zip(event_images, sides, labels)):
        axes[row, 0].imshow(base)
        axes[row, 0].set_ylabel(label, fontsize=8)
        if row == 0:
            axes[row, 0].set_title("ON/OFF events", fontsize=9)
        stage_start = 1
        if has_rgb:
            axes[row, 1].imshow(rgb_images[row])
            if row == 0:
                axes[row, 1].set_title("Paired RGB", fontsize=9)
            stage_start = 2
        for column, (name, maps) in enumerate(zip(stage_names, projected_stages), start=stage_start):
            axes[row, column].imshow(maps[row].reshape(side, side, 3), interpolation="bilinear")
            if row == 0:
                variance = 100.0 * sum(explained[name])
                axes[row, column].set_title(f"{name}\nPC1--3: {variance:.1f}%", fontsize=8.5)
        for panel in axes[row]:
            panel.set_xticks([]); panel.set_yticks([])
    fig.suptitle(f"{dataset_name}: DINO-style patch-feature PCA mapped to RGB", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "patch_pca_rgb.png", bbox_inches="tight")
    fig.savefig(output_dir / "patch_pca_rgb.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"samples": sample_ids, "explained_variance": explained}


@torch.inference_mode()
def plot_attention(model, dataset: Dataset, indices: list[int], classes: list[int], count: int,
                   device: torch.device, output_dir: Path, dataset_name: str):
    selected = select_visual_indices(dataset, indices, classes, count)
    capture = QKCapture(model)
    columns = 1 + len(model.blocks)
    fig, axes = plt.subplots(len(selected), columns, figsize=(2.25 * columns, 2.05 * len(selected)), dpi=220)
    axes = np.atleast_2d(axes)
    report = []
    for row, index in enumerate(selected):
        events, label, *_ = dataset[index]
        capture.clear()
        model(events.unsqueeze(0).to(device))
        base = event_rgb(events)
        axes[row, 0].imshow(base)
        axes[row, 0].set_ylabel(class_name(dataset, index, dataset_name), fontsize=8)
        if row == 0: axes[row, 0].set_title("ON/OFF events", fontsize=9)
        maps = []
        for block_index, block in enumerate(model.blocks):
            heat = token_importance(capture.values[block_index]["q"], capture.values[block_index]["k"], block.attn.heads)
            heat = F.interpolate(heat[None, None], size=events.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
            maps.append(heat)
            axes[row, block_index + 1].imshow(base)
            axes[row, block_index + 1].imshow(heat, cmap="inferno", alpha=0.62, vmin=0, vmax=1)
            if row == 0: axes[row, block_index + 1].set_title(f"Block {block_index + 1}", fontsize=9)
        report.append({"sample_id": sample_id(dataset, index), "label": int(label),
                       "block_peak_affinity": [float(value.max()) for value in maps]})
        for panel in axes[row]: panel.set_xticks([]); panel.set_yticks([])
    capture.close()
    fig.suptitle(f"{dataset_name}: spike-attention affinity over event evidence", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "spike_attention_heatmaps.png", bbox_inches="tight")
    fig.savefig(output_dir / "spike_attention_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)
    return report


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.project_root / "code" / "core"))
    import data as data_module
    import model as model_module

    print(json.dumps({"model_file": str(Path(model_module.__file__).resolve())}), flush=True)
    device = torch.device("cuda", args.gpu); torch.cuda.set_device(args.gpu)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(model_module, checkpoint, device)
    dataset, split_indices = build_dataset(args, data_module, checkpoint["args"])
    indices, classes = choose_balanced(dataset, split_indices, args.max_classes, args.samples_per_class, args.seed)
    features, labels = extract_layers(model, dataset, indices, args.batch_size, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = "CIFAR10-RCLS" if args.dataset == "cifar10rcls" else "N-Caltech101"
    visual_indices = select_visual_indices(
        dataset, indices, classes, args.heatmap_samples,
    )
    report = {
        "dataset": args.dataset, "checkpoint": str(args.checkpoint), "samples": len(indices),
        "classes": classes, "model_file": str(Path(model_module.__file__).resolve()),
        "tsne": plot_layerwise_tsne(features, labels, args.output_dir, dataset_name),
        "patch_pca_rgb": plot_patch_pca_rgb(
            model, dataset, visual_indices, device, args.output_dir, dataset_name, args.rgb_root,
        ),
        "attention_samples": plot_attention(model, dataset, indices, classes, args.heatmap_samples,
                                             device, args.output_dir, dataset_name),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
