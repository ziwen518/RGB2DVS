from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def stratified_split_indices(labels: torch.Tensor, validation_samples: int, seed: int):
    classes = labels.unique(sorted=True).tolist()
    per_class, remainder = divmod(validation_samples, len(classes))
    generator = torch.Generator().manual_seed(seed)
    fit_indices, validation_indices = [], []
    for offset, label in enumerate(classes):
        candidates = torch.where(labels == label)[0]
        candidates = candidates[torch.randperm(len(candidates), generator=generator)]
        count = per_class + int(offset < remainder)
        validation_indices.extend(candidates[:count].tolist())
        fit_indices.extend(candidates[count:].tolist())
    return torch.tensor(sorted(fit_indices)), sorted(validation_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-cache", required=True)
    parser.add_argument("--rgb-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2027)
    args = parser.parse_args()
    event_payload = torch.load(args.event_cache, map_location="cpu")
    rgb_payload = torch.load(args.rgb_cache, map_location="cpu")
    event_ids = [str(value) for value in event_payload["ids"]]
    rgb_by_id = {
        str(key): F.normalize(value.float().mean(0), dim=-1).double()
        for key, value in zip(rgb_payload["ids"], rgb_payload["prefix"])
    }
    x = F.normalize(event_payload["prefix"].float().mean(1), dim=-1).double()
    y = torch.stack([rgb_by_id[key] for key in event_ids])
    groups: dict[str, list[int]] = {}
    for index, sample_id in enumerate(event_ids):
        parts = sample_id.replace("\\", "/").split("/")
        label = parts[1] if parts[0] in {"train", "test"} and len(parts) > 2 else parts[0]
        groups.setdefault(label, []).append(index)
    if "labels" not in event_payload:
        raise KeyError("event cache must include labels for the shared stratified split")
    labels = torch.as_tensor(event_payload["labels"]).long()
    fit_indices, validation_indices = stratified_split_indices(
        labels, args.validation_samples, args.split_seed
    )
    validation_set = set(validation_indices)
    fit_x, fit_y = x[fit_indices], y[fit_indices]
    x_mean = fit_x.mean(0, keepdim=True)
    y_mean = fit_y.mean(0, keepdim=True)
    centered_fit_x = fit_x - x_mean
    centered_fit_y = fit_y - y_mean
    gram = centered_fit_x.t() @ centered_fit_x
    weight = torch.linalg.solve(
        gram + args.ridge * torch.eye(gram.shape[0], dtype=gram.dtype),
        centered_fit_x.t() @ centered_fit_y,
    )
    prediction = ((x - x_mean) @ weight + y_mean).float()
    observable = F.normalize(prediction, dim=-1)
    residual = y.float() - prediction
    class_names = sorted(groups)
    residual_prototypes = F.normalize(torch.stack([
        residual[torch.tensor([
            index for index in groups[label] if index not in validation_set
        ])].mean(dim=0)
        for label in class_names
    ]), dim=-1)
    output = {
        "ids": event_ids,
        "observable": observable,
        "residual_prototypes": residual_prototypes,
        "residual_class_names": class_names,
        "event_mean": x_mean.float(),
        "rgb_mean": y_mean.float(),
        "weight": weight.float(),
        "ridge": args.ridge,
        "fit_samples": len(fit_indices),
        "validation_samples": len(validation_indices),
        "fit_ids": [event_ids[index] for index in fit_indices.tolist()],
        "validation_ids": [event_ids[index] for index in validation_indices],
        "split_seed": args.split_seed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)


if __name__ == "__main__":
    main()
