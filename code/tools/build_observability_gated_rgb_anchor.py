"""Build a label-free semantic anchor gated by event observability."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def cache_values(payload: dict) -> torch.Tensor:
    values = payload.get("transported", payload.get("observable"))
    if values is not None:
        return values.float()
    prefix = payload.get("prefix")
    if prefix is None:
        raise KeyError("cache has neither target values nor prefix")
    return prefix[:, -1].float()


def indexed_values(payload: dict, wanted: list[str]) -> torch.Tensor:
    values = cache_values(payload)
    ids = list(map(str, payload["ids"]))
    if len(ids) != len(values):
        raise ValueError("cache IDs and values have different lengths")
    by_id = {sample_id: index for index, sample_id in enumerate(ids)}
    missing = set(wanted) - set(by_id)
    if missing:
        raise KeyError(f"cache misses {len(missing)} requested IDs")
    return torch.stack([values[by_id[sample_id]] for sample_id in wanted])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--observable-cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-observable-weight", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 <= args.max_observable_weight <= 1.0:
        raise ValueError("--max-observable-weight must be in [0, 1]")

    manifest_bytes = args.split_manifest.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    fit_ids = list(map(str, manifest["train_ids"]))
    validation_ids = list(map(str, manifest["validation_ids"]))
    test_ids = list(map(str, manifest.get("test_ids", [])))
    split_sets = [set(fit_ids), set(validation_ids), set(test_ids)]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train, validation, and test IDs must be disjoint")
    target_ids = fit_ids + validation_ids

    # Source caches may contain labels for historical diagnostics; this builder
    # intentionally never reads those fields.
    rgb_payload = torch.load(args.rgb_cache, map_location="cpu", weights_only=False)
    observable_payload = torch.load(
        args.observable_cache, map_location="cpu", weights_only=False
    )
    rgb = F.normalize(indexed_values(rgb_payload, target_ids), dim=-1)
    observable = F.normalize(indexed_values(observable_payload, target_ids), dim=-1)
    agreement = F.cosine_similarity(rgb, observable, dim=-1).clamp(0.0, 1.0)
    observable_weight = args.max_observable_weight * agreement
    target = F.normalize(
        rgb + observable_weight.unsqueeze(-1) * observable, dim=-1
    )

    output = {
        "ids": target_ids,
        "transported": target,
        "fit_ids": fit_ids,
        "validation_ids": validation_ids,
        "excluded_test_ids": test_ids,
        "target_type": "observability_gated_rgb_anchor",
        "source": "paired RGB-DINO gated by label-free event observable agreement",
        "max_observable_weight": args.max_observable_weight,
        "mean_observable_weight": float(observable_weight.mean()),
        "mean_rgb_observable_cosine": float(
            F.cosine_similarity(rgb, observable, dim=-1).mean()
        ),
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "labels_used": False,
        "class_prototypes_used": False,
        "test_used_for_estimation": False,
        "test_target_emitted": False,
        "fit_only_estimation": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({
        "output": str(args.output), "fit_ids": len(fit_ids),
        "validation_ids": len(validation_ids), "excluded_test_ids": len(test_ids),
        "target_ids": len(target_ids), "labels_used": False,
        "test_used_for_estimation": False, "test_target_emitted": False,
        "mean_observable_weight": output["mean_observable_weight"],
        "mean_rgb_observable_cosine": output["mean_rgb_observable_cosine"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
