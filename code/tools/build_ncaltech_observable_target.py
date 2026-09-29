"""Fit an event-statistics -> RGB-DINO observable relay for N-Caltech101.

This is a training-only, fit-split ridge relay. It uses no pixel correspondence;
the RGB target is paired only by the official event basename.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def split_indices(ids, split):
    payload = json.loads(split.read_text(encoding="utf-8"))
    groups = {str(x): name for name in ("train", "validation", "test") for x in payload.get(name, [])}
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event-cache", type=Path, required=True)
    p.add_argument("--rgb-cache", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--validation-samples", type=int, default=900)
    args = p.parse_args()
    event = torch.load(args.event_cache, map_location="cpu", weights_only=False)
    rgb = torch.load(args.rgb_cache, map_location="cpu", weights_only=False)
    ids = [str(x) for x in event["ids"]]
    rgb_by_id = {str(k): F.normalize(v.float().mean(0), dim=-1) for k, v in zip(rgb["ids"], rgb["prefix"])}
    missing = set(ids) - set(rgb_by_id)
    if missing:
        raise KeyError(f"RGB cache missing {len(missing)} event IDs")
    labels = torch.as_tensor(event["labels"]).long()
    groups = split_indices(ids, args.split)
    fit = torch.tensor([i for i, x in enumerate(ids) if groups.get(x) == "train"], dtype=torch.long)
    if len(fit) == 0:
        raise ValueError("split has no train IDs")
    x = F.normalize(event["prefix"].float().mean(1), dim=-1).double()
    y = torch.stack([rgb_by_id[xid] for xid in ids]).double()
    xm, ym = x[fit].mean(0, keepdim=True), y[fit].mean(0, keepdim=True)
    xc, yc = x - xm, y - ym
    gram = xc[fit].t() @ xc[fit]
    w = torch.linalg.solve(gram + args.ridge * torch.eye(gram.shape[1], dtype=gram.dtype), xc[fit].t() @ yc[fit])
    pred = F.normalize(((xc @ w) + ym).float(), dim=-1)
    confidence = F.cosine_similarity(pred, y.float(), dim=-1).clamp(0, 1)
    payload = {
        "ids": ids, "observable": pred, "transported": pred,
        "observable_confidence": confidence, "labels": labels,
        "fit_ids": [ids[i] for i in fit.tolist()],
        "validation_ids": [x for x in ids if groups.get(x) == "validation"],
        "test_ids": [x for x in ids if groups.get(x) == "test"],
        "source": "N-Caltech101 event-statistics ridge relay to paired RGB-DINO",
        "ridge": args.ridge, "fit_only_estimation": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"samples": len(ids), "fit": len(payload["fit_ids"]), "validation": len(payload["validation_ids"]), "test": len(payload["test_ids"]), "mean_confidence": float(confidence.mean())}), flush=True)


if __name__ == "__main__":
    main()
