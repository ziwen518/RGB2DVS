from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-cache", type=Path, required=True)
    parser.add_argument("--rgb-cache", type=Path, required=True)
    parser.add_argument("--observable-target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    event = torch.load(args.event_cache, map_location="cpu", weights_only=False)
    rgb = torch.load(args.rgb_cache, map_location="cpu", weights_only=False)
    observable = torch.load(
        args.observable_target, map_location="cpu", weights_only=False
    )
    event_source_ids = [str(sample_id) for sample_id in event["ids"]]
    observable_ids = [str(sample_id) for sample_id in observable["ids"]]
    rgb_by_id = {
        str(sample_id): feature.float().mean(0)
        for sample_id, feature in zip(rgb["ids"], rgb["prefix"])
    }
    observable_by_id = {
        str(sample_id): feature.float()
        for sample_id, feature in zip(
            observable["ids"], observable["transported"]
        )
    }
    if not set(observable_ids).issubset(event_source_ids) or not set(
        observable_ids
    ).issubset(rgb_by_id):
        raise ValueError("Observable target IDs are not covered by Event/RGB caches")
    # The Event feature cache can cover a larger protocol. The trained
    # observable target owns the fit/validation split used by downstream KD.

    event_features = event.get("observable")
    if event_features is None:
        event_features = event["prefix"].float().mean(1)
    event_by_id = {
        sample_id: feature.float()
        for sample_id, feature in zip(event_source_ids, event_features)
    }
    event_ids = observable_ids
    targets = {
        "event": torch.stack([event_by_id[sample_id] for sample_id in event_ids]),
        "full_rgb": torch.stack([rgb_by_id[sample_id] for sample_id in event_ids]),
        "observable": torch.stack([
            observable_by_id[sample_id] for sample_id in event_ids
        ]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target_type, features in targets.items():
        payload = {
            "ids": event_ids,
            "fit_ids": list(observable["fit_ids"]),
            "validation_ids": list(observable["validation_ids"]),
            "split_manifest_fingerprint": observable.get(
                "split_manifest_fingerprint",
                event.get("split_manifest_fingerprint"),
            ),
            "fit_only_estimation": target_type == "observable",
            "test_used": False,
            "transported": F.normalize(features, dim=-1),
            "target_type": target_type,
        }
        path = args.output_dir / f"{target_type}.pt"
        torch.save(payload, path)
        print(
            {
                "path": str(path),
                "target_type": target_type,
                "samples": len(event_ids),
                "dimension": features.shape[-1],
                "finite": bool(torch.isfinite(features).all()),
            }
        )


if __name__ == "__main__":
    main()
