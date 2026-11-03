from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--band-ends", type=int, nargs="+", default=[16, 64, 128],
        help="exclusive cumulative PCA band endpoints",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.band_ends or any(end <= 0 for end in args.band_ends):
        raise ValueError("PCA band endpoints must be positive")
    if args.band_ends != sorted(set(args.band_ends)):
        raise ValueError("PCA band endpoints must be strictly increasing")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
    fit_ids = [str(sample_id) for sample_id in manifest["train_ids"]]
    payload = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    values = payload.get("transported", payload.get("observable"))
    if values is None:
        raise KeyError("target cache has neither transported nor observable features")
    target_by_id = {
        str(sample_id): value.float()
        for sample_id, value in zip(payload["ids"], values)
    }
    missing = set(fit_ids) - set(target_by_id)
    if missing:
        raise KeyError(f"target cache is missing {len(missing)} fit IDs")

    matrix = torch.stack([target_by_id[sample_id] for sample_id in fit_ids])
    if matrix.ndim != 2:
        raise ValueError("observable targets must have shape [samples, features]")
    matrix = F.normalize(matrix.float(), dim=-1)
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    covariance = centered.t().matmul(centered) / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    components = eigenvectors[:, order].t().contiguous()

    component_count = args.band_ends[-1]
    if component_count > components.shape[0]:
        raise ValueError("requested PCA components exceed the feature dimension")
    explained_ratio = eigenvalues / eigenvalues.sum().clamp_min(1.0e-12)
    starts = [0, *args.band_ends[:-1]]
    bands = [[start, end] for start, end in zip(starts, args.band_ends)]
    id_hash = hashlib.sha256("\n".join(fit_ids).encode("utf-8")).hexdigest()
    result = {
        "mean": mean.cpu(),
        "components": components[:component_count].cpu(),
        "scales": eigenvalues[:component_count].sqrt().clamp_min(1.0e-4).cpu(),
        "explained_variance": eigenvalues[:component_count].cpu(),
        "explained_variance_ratio": explained_ratio[:component_count].cpu().tolist(),
        "bands": bands,
        "metadata": {
            "target_cache": str(args.target_cache.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "fit_samples": len(fit_ids),
            "feature_dim": int(matrix.shape[1]),
            "components": component_count,
            "fit_ids_sha256": id_hash,
            "target_normalization": "l2",
            "labels_used": False,
            "validation_used": False,
            "test_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(result, temporary)
    temporary.replace(args.output)
    summary = {
        **result["metadata"],
        "output": str(args.output.resolve()),
        "bands": bands,
        "explained_variance_ratio_by_band": [
            float(explained_ratio[start:end].sum()) for start, end in bands
        ],
        "explained_variance_ratio_total": float(
            explained_ratio[:component_count].sum()
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
