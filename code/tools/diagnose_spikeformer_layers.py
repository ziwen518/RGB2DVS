from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def centered_linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float() - first.float().mean(0, keepdim=True)
    second = second.float() - second.float().mean(0, keepdim=True)
    cross = first.t() @ second
    denominator = torch.linalg.norm(first.t() @ first) * torch.linalg.norm(second.t() @ second)
    return float(cross.square().sum() / denominator.clamp_min(1.0e-12))


def effective_rank(feature: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(feature.float() - feature.float().mean(0, keepdim=True))
    probability = singular / singular.sum().clamp_min(1.0e-12)
    return float(torch.exp(-(probability * probability.clamp_min(1.0e-12).log()).sum()))


def pairwise_cka(features: dict[str, torch.Tensor]) -> dict[str, object]:
    names = list(features)
    return {
        "layers": names,
        "matrix": [
            [centered_linear_cka(features[first], features[second]) for second in names]
            for first in names
        ],
    }


def knn(bank, bank_y, query, query_y, k=10) -> float:
    neighbors = (F.normalize(query, dim=-1) @ F.normalize(bank, dim=-1).t()).topk(k, dim=-1).indices
    prediction = torch.mode(bank_y[neighbors], dim=-1).values
    return float(prediction.eq(query_y).float().mean())


@torch.inference_mode()
def extract(model, loader, device):
    model.eval(); layer_batches, membrane_batches, core, signed, labels, ids = [], [], [], [], [], []
    for events, target, _, sample_ids in loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                events.to(device, non_blocking=True),
                return_layers=True, return_membranes=True,
            )
        layer_batches.append(output["layer_embeddings"].cpu())
        membrane_batches.append(output["membrane_embeddings"].cpu())
        core.append(output["core_embedding"].cpu())
        signed.append(output["embedding"].cpu())
        labels.append(torch.as_tensor(target)); ids.extend(map(str, sample_ids))
    layers = torch.cat(layer_batches)
    features = {"stem": layers[:, 0]}
    features.update({f"block_{index}": layers[:, index] for index in range(1, layers.shape[1])})
    membranes = torch.cat(membrane_batches)
    features.update({
        f"block_{index + 1}_membrane": membranes[:, index]
        for index in range(membranes.shape[1])
    })
    features["core"] = torch.cat(core)
    features["signed_readout"] = torch.cat(signed)
    if "block_2" in features and "block_4" in features:
        features["block_2_4_mean"] = F.normalize(
            features["block_2"] + features["block_4"], dim=-1
        )
        features["block_2_4_concat"] = torch.cat(
            [features["block_2"], features["block_4"]], dim=-1
        )
    return features, torch.cat(labels), ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--full-dataset", action="store_true",
        help="Use every sample in data_root/train and data_root/test instead of the manifest.",
    )
    args = parser.parse_args()
    # Resolve the isolated rgb2dvs layout before any legacy project modules.
    script_dir = Path(__file__).resolve().parent
    core_dir = script_dir.parent / "core"
    if not core_dir.is_dir():
        core_dir = script_dir
    sys.path.insert(0, str(core_dir))
    sys.path.insert(1, str(script_dir))
    sys.path.insert(2, str(args.project_root))
    from data import make_event_dataset
    import model as model_module
    from model import PureSpikeFormer

    print(json.dumps({"model_file": str(Path(model_module.__file__).resolve())}), flush=True)

    torch.cuda.set_device(args.gpu); device = torch.device("cuda", args.gpu)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    dataset = make_event_dataset(
        "motion", args.data_root, True, steps=int(config["steps"]),
        size=int(config["size"]), two_views=False,
        frame_mode=config["frame_mode"], preload_frames=True,
    )
    if args.full_dataset:
        validation_dataset = make_event_dataset(
            "motion", args.data_root, False, steps=int(config["steps"]),
            size=int(config["size"]), two_views=False,
            frame_mode=config["frame_mode"], preload_frames=True,
        )
        fit_loader = DataLoader(
            dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
        validation_loader = DataLoader(
            validation_dataset, args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
    else:
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        id_to_index = {
            str(path.relative_to(dataset.root)).replace("\\", "/"): index
            for index, (path, _, _) in enumerate(dataset.samples)
        }
        fit_ids = list(map(str, manifest["train_ids"])); validation_ids = list(map(str, manifest["validation_ids"]))
        fit_loader = DataLoader(
            IndexedDataset(dataset, [id_to_index[key] for key in fit_ids]),
            args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
        validation_loader = DataLoader(
            IndexedDataset(dataset, [id_to_index[key] for key in validation_ids]),
            args.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
    input_channels = int(dataset[0][0].shape[1])
    model = PureSpikeFormer(
        in_channels=input_channels, dim=int(config["dim"]), depth=int(config["depth"]),
        heads=int(config["heads"]), patch_size=int(config["patch_size"]),
        threshold=float(config["threshold"]),
        use_cls_token=bool(config.get("use_cls_token", False)), norm="bntt",
        image_size=int(config["size"]), temporal_readout="learned",
        local_stem=not bool(config.get("pyramid_stem", False)),
        pyramid_stem=bool(config.get("pyramid_stem", False)),
        use_positional_bias=True, attention_mode="normalized",
        signed_readout=bool(config.get("signed_readout", True)),
        multidepth_readout=bool(config.get("multidepth_readout", False)),
        multidepth_readout_layers=int(config.get("multidepth_readout_layers", 3)),
        block_spiking=config.get("block_spiking", "full_respike"),
        hybrid_attention=bool(config.get("hybrid_attention", False)),
        hybrid_attention_suffix=int(config.get("hybrid_attention_suffix", 0)),
        qkv_temporal_mode=config.get("qkv_temporal_mode", "standard"),
        ternary_threshold=float(config.get("ternary_threshold", 0.5)),
        continuous=bool(config.get("continuous_relaxation", False)),
        temporal_steps=int(config["steps"]),
        population_bins=int(config.get("population_bins", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    fit_features, fit_y, extracted_fit_ids = extract(model, fit_loader, device)
    validation_features, validation_y, extracted_validation_ids = extract(model, validation_loader, device)
    fit_teacher = validation_teacher = None
    if args.target_cache is not None:
        teacher_payload = torch.load(args.target_cache, map_location="cpu", weights_only=False)
        teacher_by_id = {
            str(key): F.normalize(value.float(), dim=-1)
            for key, value in zip(teacher_payload["ids"], teacher_payload["transported"])
        }
        fit_teacher = torch.stack([teacher_by_id[key] for key in extracted_fit_ids])
        validation_teacher = torch.stack([teacher_by_id[key] for key in extracted_validation_ids])
    results = []
    for name in fit_features:
        train_x = fit_features[name].numpy(); validation_x = validation_features[name].numpy()
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        )
        probe.fit(train_x, fit_y.numpy())
        results.append({
            "layer": name,
            "knn": knn(fit_features[name], fit_y, validation_features[name], validation_y),
            "linear_probe_train": float(probe.score(train_x, fit_y.numpy())),
            "linear_probe_validation": float(probe.score(validation_x, validation_y.numpy())),
            "effective_rank": effective_rank(fit_features[name]),
            "embedding_std": float(fit_features[name].std(dim=0).mean()),
            "cka_fit_observable": (
                centered_linear_cka(fit_features[name], fit_teacher)
                if fit_teacher is not None else None
            ),
            "cka_validation_observable": (
                centered_linear_cka(validation_features[name], validation_teacher)
                if validation_teacher is not None else None
            ),
        })
    output = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "model_file": str(Path(model_module.__file__).resolve()),
        "fit_samples": len(fit_y), "validation_samples": len(validation_y),
        "full_dataset": bool(args.full_dataset),
        "layers": results,
        "pairwise_cka_fit": pairwise_cka(fit_features),
        "pairwise_cka_validation": pairwise_cka(validation_features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps(output, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
