from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a strict train-only view of an existing patch target cache."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_ids(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {args.output}")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
    train_ids = list(map(str, manifest["train_ids"]))
    validation_ids = set(map(str, manifest.get("validation_ids", [])))
    test_ids = set(map(str, manifest.get("test_ids", [])))
    if set(train_ids) & (validation_ids | test_ids):
        raise ValueError("split manifest is not disjoint")

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    source_ids = list(map(str, source["ids"]))
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source cache contains duplicate IDs")
    index = {sample_id: row for row, sample_id in enumerate(source_ids)}
    missing = set(train_ids) - set(index)
    if missing:
        raise KeyError(f"source cache is missing {len(missing)} train IDs")
    rows = torch.tensor([index[sample_id] for sample_id in train_ids], dtype=torch.long)

    output: dict[str, object] = {}
    per_sample_keys: list[str] = []
    for key, value in source.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == len(source_ids):
            output[key] = value.index_select(0, rows)
            per_sample_keys.append(str(key))
        elif key not in {"ids", "fit_ids", "validation_ids", "test_ids"}:
            output[key] = value
    output["ids"] = train_ids
    output["fit_ids"] = train_ids
    output["validation_ids"] = []
    output["test_ids"] = []
    output["strict_audit"] = {
        "fit_only_estimation": True,
        "fit_only_targets_emitted": True,
        "labels_used": False,
        "class_prototypes_used": False,
        "validation_used_for_estimation": False,
        "validation_target_emitted": False,
        "test_used_for_estimation": False,
        "test_target_emitted": False,
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "test_count": len(test_ids),
        "train_ids_sha256": sha256_ids(train_ids),
        "per_sample_keys": sorted(per_sample_keys),
        "source_cache": str(args.source),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)

    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    emitted_ids = list(map(str, reloaded["ids"]))
    audit: dict[str, object] = {
        "output": str(args.output),
        "ids": len(emitted_ids),
        "validation_overlap": len(set(emitted_ids) & validation_ids),
        "test_overlap": len(set(emitted_ids) & test_ids),
        **reloaded["strict_audit"],
    }
    for key, value in reloaded.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == len(emitted_ids):
            audit[f"{key}_shape"] = list(value.shape)
    if emitted_ids != train_ids or audit["validation_overlap"] or audit["test_overlap"]:
        raise RuntimeError("strict cache post-write audit failed")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
