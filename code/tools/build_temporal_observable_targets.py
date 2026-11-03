from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> int:
    p = argparse.ArgumentParser(description="Build causal temporal observable targets.")
    p.add_argument("--observable", type=Path, required=True)
    p.add_argument("--event-teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gamma", type=float, default=2.0)
    args = p.parse_args()
    observable = torch.load(args.observable, map_location="cpu", weights_only=False)
    event = torch.load(args.event_teacher, map_location="cpu", weights_only=False)
    event_by_id = {str(k): v.float() for k, v in zip(event["ids"], event["prefix"])}
    ids = [str(k) for k in observable["ids"]]
    rgb = F.normalize(observable.get("transported", observable.get("observable")).float(), dim=-1)
    event_prefix = torch.stack([event_by_id[k] for k in ids])
    event_prefix = F.normalize(event_prefix, dim=-1)
    steps = event_prefix.shape[1]
    fractions = torch.arange(1, steps + 1, dtype=torch.float32) / steps
    fractions = fractions.pow(args.gamma).view(1, steps, 1)
    temporal = F.normalize(
        (1.0 - fractions) * event_prefix + fractions * rgb[:, None], dim=-1,
    )
    output = {
        "ids": ids,
        "temporal_targets": temporal,
        "steps": steps,
        "gamma": args.gamma,
        "source": "causal event-teacher prefixes to RGB-observable semantic schedule",
        "fit_ids": observable.get("fit_ids", []),
        "validation_ids": observable.get("validation_ids", []),
        "test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({"output": str(args.output), "shape": list(temporal.shape), "gamma": args.gamma}))
    return 0


if __name__ == "__main__":
    main()
