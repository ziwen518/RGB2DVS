"""Audit label-free target caches with labels used for kNN evaluation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def sample_label(sample_id: str) -> str:
    return str(sample_id).replace("\\", "/").split("/")[-2]


def target_values(payload: dict) -> torch.Tensor:
    values = payload.get("transported", payload.get("observable"))
    if values is not None:
        return values.float()
    prefix = payload.get("prefix")
    if prefix is None:
        raise KeyError("cache has neither target features nor prefix")
    return prefix[:, -1].float()


def knn_accuracy(
    train_x: torch.Tensor, train_labels: list[str],
    validation_x: torch.Tensor, validation_labels: list[str], k: int,
) -> float:
    train_x = F.normalize(train_x, dim=-1)
    validation_x = F.normalize(validation_x, dim=-1)
    classes = {
        name: index for index, name in enumerate(
            sorted(set(train_labels) | set(validation_labels))
        )
    }
    train_y = torch.tensor([classes[name] for name in train_labels])
    validation_y = torch.tensor([classes[name] for name in validation_labels])
    correct = 0
    for start in range(0, len(validation_x), 128):
        neighbors = (validation_x[start:start + 128] @ train_x.T).topk(k, dim=-1).indices
        prediction = torch.mode(train_y[neighbors], dim=-1).values
        correct += int(prediction.eq(validation_y[start:start + 128]).sum())
    return correct / max(len(validation_x), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    values = target_values(payload)
    ids = list(map(str, payload["ids"]))
    if len(ids) != len(values):
        raise ValueError("target values and IDs have different lengths")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
    train_ids = list(map(str, manifest["train_ids"]))
    validation_ids = list(map(str, manifest["validation_ids"]))
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation IDs overlap")
    by_id = {sample_id: values[index] for index, sample_id in enumerate(ids)}
    missing = (set(train_ids) | set(validation_ids)) - set(by_id)
    if missing:
        raise KeyError(f"cache misses {len(missing)} requested IDs")
    accuracy = knn_accuracy(
        torch.stack([by_id[key] for key in train_ids]),
        [sample_label(key) for key in train_ids],
        torch.stack([by_id[key] for key in validation_ids]),
        [sample_label(key) for key in validation_ids],
        args.knn_k,
    )
    print(json.dumps({
        "cache": str(args.cache), "train_samples": len(train_ids),
        "validation_samples": len(validation_ids), "knn_k": args.knn_k,
        "validation_knn": accuracy, "labels_used_for_evaluation_only": True,
        "cache_label_field_ignored": "labels" in payload,
        "test_used": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
