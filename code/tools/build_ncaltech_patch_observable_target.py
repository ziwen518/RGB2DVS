"""Fit an alignment-free patch-wise event->RGB observable relay.

Patch indices are treated as an unordered set.  The relay is fit independently
per semantic slot on the train split; distillation later matches patch affinity
matrices, so no RGB/event pixel correspondence is assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--event-cache", type=Path, required=True)
    p.add_argument("--rgb-cache", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ridge", type=float, default=10.0)
    args = p.parse_args()
    event = torch.load(args.event_cache, map_location="cpu", weights_only=False)
    rgb = torch.load(args.rgb_cache, map_location="cpu", weights_only=False)
    split = json.loads(args.split.read_text(encoding="utf-8-sig"))
    event_by_id = {str(k): v.float() for k, v in zip(event["ids"], event["patches"])}
    rgb_by_id = {str(k): v.float() for k, v in zip(rgb["ids"], rgb["patches"])}
    ids = [str(x) for x in event["ids"]]
    fit_ids = set(map(str, split["train"]))
    fit = torch.tensor([i for i, key in enumerate(ids) if key in fit_ids], dtype=torch.long)
    if fit.numel() == 0:
        raise ValueError("split has no train IDs")
    x = torch.stack([event_by_id[key].mean(0) for key in ids])
    y = torch.stack([rgb_by_id[key].mean(0) for key in ids])
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(f"patch cache shape mismatch: event={tuple(x.shape)} rgb={tuple(y.shape)}")
    # Slot permutations are not semantically meaningful.  A global event/RGB
    # slot similarity assignment is used only to order the relay targets; the
    # loss itself remains affinity-based and alignment-free.
    xn = F.normalize(x[fit], dim=-1)
    yn = F.normalize(y[fit], dim=-1)
    slot_score = (xn[:, :, None, :] * yn[:, None, :, :]).sum(-1).mean(0)
    assignment = []
    remaining = set(range(x.shape[1]))
    for i in range(x.shape[1]):
        j = max(remaining, key=lambda k: float(slot_score[i, k]))
        assignment.append(j); remaining.remove(j)
    y = y[:, assignment]
    pred = torch.empty_like(y)
    confidence = torch.empty((len(ids), y.shape[1]))
    for slot in range(x.shape[1]):
        xs, ys = x[fit, slot].double(), y[fit, slot].double()
        xm, ym = xs.mean(0, keepdim=True), ys.mean(0, keepdim=True)
        xc, yc = xs - xm, ys - ym
        gram = xc.t() @ xc
        w = torch.linalg.solve(gram + args.ridge * torch.eye(gram.shape[1], dtype=gram.dtype), xc.t() @ yc)
        out = ((x[:, slot].double() - xm) @ w + ym).float()
        pred[:, slot] = F.normalize(out, dim=-1)
        confidence[:, slot] = F.cosine_similarity(pred[:, slot], y[:, slot], dim=-1).clamp(0, 1)
    payload = {
        "ids": ids, "tokens": pred.half(), "confidence": confidence.half(),
        "assignment": assignment, "fit_ids": list(map(str, split["train"])),
        "validation_ids": list(map(str, split["validation"])),
        "test_ids": list(map(str, split["test"])),
        "source": "fit-only per-slot event-to-RGB relay; affinity distillation, no pixel alignment",
        "ridge": args.ridge,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print({"samples": len(ids), "patches": int(x.shape[1]), "mean_confidence": float(confidence.mean())}, flush=True)


if __name__ == "__main__":
    main()
