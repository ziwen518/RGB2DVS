from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def normalized_knn(
    bank: torch.Tensor,
    bank_labels: torch.Tensor,
    query: torch.Tensor,
    query_labels: torch.Tensor,
    k: int,
) -> float:
    bank = F.normalize(bank.float(), dim=-1)
    query = F.normalize(query.float(), dim=-1)
    similarity = query @ bank.t()
    neighbors = similarity.topk(min(k, len(bank)), dim=-1).indices
    prediction = torch.mode(bank_labels[neighbors], dim=-1).values
    return float(prediction.eq(query_labels).float().mean())


def load_feature(payload: dict, index: int) -> torch.Tensor:
    key = "observable" if "observable" in payload else "prefix"
    feature = payload[key][index].float()
    return feature.mean(dim=0) if feature.ndim == 2 else feature


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-cache", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(16, 32, 64, 96, 128, 192, 256, 384))
    parser.add_argument("--ridges", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    event_payload = torch.load(args.event_cache, map_location="cpu", weights_only=False)
    rgb_payload = torch.load(args.rgb_cache, map_location="cpu", weights_only=False)
    event_ids = list(map(str, event_payload["ids"]))
    rgb_index = {str(sample_id): index for index, sample_id in enumerate(rgb_payload["ids"])}
    missing = sorted(set(event_ids) - set(rgb_index))
    if missing:
        raise ValueError(f"RGB cache misses {len(missing)} event samples")
    fit_ids = set(map(str, event_payload["fit_ids"]))
    validation_ids = set(map(str, event_payload["validation_ids"]))
    if fit_ids & validation_ids or fit_ids | validation_ids != set(event_ids):
        raise ValueError("fit/validation IDs must be disjoint and exhaustive")

    event = torch.stack([load_feature(event_payload, index) for index in range(len(event_ids))])
    rgb = torch.stack([load_feature(rgb_payload, rgb_index[sample_id]) for sample_id in event_ids])
    labels = torch.tensor([
        int(rgb_payload["labels"][rgb_index[sample_id]]) for sample_id in event_ids
    ])
    fit = torch.tensor([index for index, sample_id in enumerate(event_ids) if sample_id in fit_ids])
    validation = torch.tensor([
        index for index, sample_id in enumerate(event_ids) if sample_id in validation_ids
    ])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    event, rgb, labels = event.to(device).float(), rgb.to(device).float(), labels.to(device)
    fit, validation = fit.to(device), validation.to(device)

    event_mean = event[fit].mean(dim=0, keepdim=True)
    rgb_mean = rgb[fit].mean(dim=0, keepdim=True)
    x = event - event_mean
    y = rgb - rgb_mean
    x_fit, y_fit = x[fit], y[fit]
    gram = x_fit.t() @ x_fit
    cross = x_fit.t() @ y_fit
    identity = torch.eye(gram.shape[0], device=device, dtype=gram.dtype)
    total_validation_energy = y[validation].square().sum().clamp_min(1e-12)

    result = {
        "protocol": {
            "fit_samples": len(fit),
            "validation_samples": len(validation),
            "fit_only_subspace_estimation": True,
            "test_used": False,
            "split_manifest_fingerprint": event_payload.get("split_manifest_fingerprint"),
        },
        "baselines": {
            "event_knn": normalized_knn(
                event[fit], labels[fit], event[validation], labels[validation], args.knn_k
            ),
            "rgb_knn": normalized_knn(
                rgb[fit], labels[fit], rgb[validation], labels[validation], args.knn_k
            ),
        },
        "settings": [],
    }
    ranks = sorted(set(min(int(rank), y.shape[1]) for rank in args.ranks))
    for ridge in args.ridges:
        # Reduced-rank ridge chooses RGB directions from fit-set predictions,
        # so target-space rank is tied to event predictability rather than RGB PCA.
        coefficient = torch.linalg.solve(gram + ridge * identity, cross)
        fitted = x_fit @ coefficient
        _, singular_values, right_t = torch.linalg.svd(fitted, full_matrices=False)
        for rank in ranks:
            basis = right_t[:rank].t()
            projected_rgb = y @ basis @ basis.t() + rgb_mean
            predicted_rgb = x @ coefficient @ basis @ basis.t() + rgb_mean
            residual_rgb = y - y @ basis @ basis.t()
            prediction_error = (
                predicted_rgb[validation] - rgb[validation]
            ).square().sum()
            row = {
                "ridge": ridge,
                "rank": rank,
                "projected_rgb_knn": normalized_knn(
                    projected_rgb[fit], labels[fit], projected_rgb[validation],
                    labels[validation], args.knn_k,
                ),
                "predicted_rgb_knn": normalized_knn(
                    predicted_rgb[fit], labels[fit], predicted_rgb[validation],
                    labels[validation], args.knn_k,
                ),
                "unobservable_residual_knn": normalized_knn(
                    residual_rgb[fit], labels[fit], residual_rgb[validation],
                    labels[validation], args.knn_k,
                ),
                "prediction_rgb_cosine": float(F.cosine_similarity(
                    predicted_rgb[validation].float(), rgb[validation].float(), dim=-1
                ).mean()),
                "prediction_r2": float(1.0 - prediction_error / total_validation_energy),
                "observable_rgb_energy": float(
                    (y[validation] @ basis).square().sum() / total_validation_energy
                ),
                "retained_fitted_energy": float(
                    singular_values[:rank].square().sum()
                    / singular_values.square().sum().clamp_min(1e-12)
                ),
            }
            result["settings"].append(row)
            print(json.dumps(row), flush=True)

    result["best_predicted"] = max(
        result["settings"], key=lambda item: item["predicted_rgb_knn"]
    )
    result["best_projected"] = max(
        result["settings"], key=lambda item: item["projected_rgb_knn"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "baselines": result["baselines"],
        "best_predicted": result["best_predicted"],
        "best_projected": result["best_projected"],
        "output": str(args.output),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
