from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class AugmentedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], augment_fn=None) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        events, target, ordinal, sample_id = self.dataset[self.indices[index]]
        if self.augment_fn is not None:
            events = self.augment_fn(events)
        return events, target, ordinal, sample_id


class StudentClassifier(nn.Module):
    def __init__(self, student: nn.Module, dim: int, classes: int, representation: str) -> None:
        super().__init__()
        self.student = student
        self.head = nn.Linear(dim, classes)
        self.representation = representation

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.head(self.student(events)[self.representation])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--real-events-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 9, 18, 45])
    p.add_argument("--protocols", nargs="+", choices=["lp", "ft"], default=["lp", "ft"])
    p.add_argument("--representation", choices=["embedding", "core_embedding"], default="embedding")
    p.add_argument("--num-subsets", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr-linear", type=float, default=1.0e-3)
    p.add_argument("--lr-finetune", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--cutmix-probability", type=float, default=0.5)
    p.add_argument("--augmentation", choices=["none", "weak"], default="weak")
    p.add_argument("--real-steps", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_from_checkpoint(model_class, checkpoint: dict, device: torch.device) -> nn.Module:
    cfg = checkpoint["args"]
    model = model_class(
        in_channels=2,
        dim=int(cfg["dim"]),
        depth=int(cfg["depth"]),
        heads=int(cfg["heads"]),
        patch_size=int(cfg["patch_size"]),
        threshold=float(cfg["threshold"]),
        use_cls_token=bool(cfg.get("use_cls_token", False)),
        norm="bntt",
        image_size=int(cfg["size"]),
        temporal_readout="learned",
        local_stem=not bool(cfg.get("pyramid_stem", False)),
        pyramid_stem=bool(cfg.get("pyramid_stem", False)),
        use_positional_bias=True,
        attention_mode="normalized",
        signed_readout=bool(cfg.get("signed_readout", True)),
        multidepth_readout=bool(cfg.get("multidepth_readout", False)),
        multidepth_readout_layers=int(cfg.get("multidepth_readout_layers", 3)),
        block_spiking=cfg.get("block_spiking", "full_respike"),
        hybrid_attention=bool(cfg.get("hybrid_attention", False)),
        hybrid_attention_suffix=int(cfg.get("hybrid_attention_suffix", 0)),
        qkv_temporal_mode=cfg.get("qkv_temporal_mode", "standard"),
        continuous=bool(cfg.get("continuous_relaxation", False)),
        ternary_threshold=float(cfg.get("ternary_threshold", 0.5)),
        temporal_steps=int(cfg["steps"]),
        population_bins=int(cfg.get("population_bins", 1)),
        event_prompt=bool(cfg.get("event_prompt", False)),
        prompt_strength=float(cfg.get("prompt_strength", 0.7)),
        local_structure_mixer=bool(cfg.get("local_structure_mixer", False)),
        local_mixer_dilations=tuple(cfg.get("local_mixer_dilations", [1, 2])),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def stratified_shot_indices(dataset, shots: int, seed: int, classes: int = 10) -> list[int]:
    rng = np.random.default_rng(seed)
    labels = np.asarray([int(sample[1]) for sample in dataset.samples])
    selected: list[int] = []
    for label in range(classes):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < shots:
            raise ValueError(f"class {label} has only {len(candidates)} samples, requested {shots}")
        selected.extend(rng.choice(candidates, size=shots, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def cutmix(events: torch.Tensor, targets: torch.Tensor, probability: float):
    if events.shape[0] < 2 or random.random() >= probability:
        return events, targets, None, 1.0
    permutation = torch.randperm(events.shape[0], device=events.device)
    lam = float(np.random.beta(1.0, 1.0))
    height, width = events.shape[-2:]
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = random.randrange(height), random.randrange(width)
    y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)
    mixed = events.clone()
    mixed[..., y1:y2, x1:x2] = events[permutation, ..., y1:y2, x1:x2]
    actual = 1.0 - ((y2 - y1) * (x2 - x1) / float(height * width))
    return mixed, targets, targets[permutation], actual


@torch.inference_mode()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for events, targets, *_ in loader:
        events = events.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(events)
        correct += int(logits.argmax(dim=-1).eq(targets).sum())
        total += int(targets.numel())
    return correct / max(total, 1)


def train_one(
    classifier: StudentClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    protocol: str,
    epochs: int,
    eval_every: int,
    lr: float,
    weight_decay: float,
    cutmix_probability: float,
) -> dict:
    freeze_student = protocol == "lp"
    for parameter in classifier.student.parameters():
        parameter.requires_grad_(not freeze_student)
    parameters = classifier.head.parameters() if freeze_student else classifier.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        classifier.train()
        if freeze_student:
            # Strict LP: freeze both parameters and normalization buffers.
            classifier.student.eval()
        total_loss = 0.0
        samples = 0
        for events, targets, *_ in train_loader:
            events = events.to(device, non_blocking=True)
            targets = torch.as_tensor(targets, device=device)
            events, targets_a, targets_b, lam = cutmix(events, targets, cutmix_probability)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = classifier(events)
                loss = F.cross_entropy(logits, targets_a)
                if targets_b is not None:
                    loss = lam * loss + (1.0 - lam) * F.cross_entropy(logits, targets_b)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            samples += int(targets.numel())
        scheduler.step()
        should_evaluate = epoch == epochs or epoch % eval_every == 0
        row = {"epoch": epoch, "train_loss": total_loss / max(samples, 1)}
        if should_evaluate:
            heldout = accuracy(classifier, test_loader, device)
            row["heldout_accuracy"] = heldout
            if heldout > best_accuracy:
                best_accuracy = heldout
                best_epoch = epoch
            print(
                f"protocol={protocol} epoch={epoch}/{epochs} "
                f"loss={row['train_loss']:.6f} heldout={heldout:.4f} "
                f"best={best_accuracy:.4f}@{best_epoch}",
                flush=True,
            )
        history.append(row)
    final_accuracy = accuracy(classifier, test_loader, device)
    return {
        "final_accuracy": final_accuracy,
        "paper_comparable_best_heldout_accuracy": best_accuracy,
        "paper_comparable_best_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def aggregate(records: list[dict], shots: list[int], protocols: list[str]) -> dict:
    result: dict[str, dict] = {}
    for protocol in protocols:
        result[protocol] = {}
        for shot in shots:
            selected = [r for r in records if r["protocol"] == protocol and r["shots"] == shot]
            final = np.asarray([r["final_accuracy"] for r in selected], dtype=np.float64)
            best = np.asarray(
                [r["paper_comparable_best_heldout_accuracy"] for r in selected], dtype=np.float64,
            )
            result[protocol][str(shot)] = {
                "runs": len(selected),
                "final_mean": float(final.mean()) if len(final) else None,
                "final_std": float(final.std(ddof=1)) if len(final) > 1 else 0.0 if len(final) else None,
                "best_heldout_mean": float(best.mean()) if len(best) else None,
                "best_heldout_std": float(best.std(ddof=1)) if len(best) > 1 else 0.0 if len(best) else None,
            }
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = args.project_root / "code" / "core"
    sys.path.insert(0, str(core))
    import model as model_module
    from data import augment_events, make_event_dataset
    from model import PureSpikeFormer

    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(args.gpu)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    if int(cfg["steps"]) != args.real_steps:
        raise ValueError(
            f"checkpoint was trained with {cfg['steps']} steps; requested real steps={args.real_steps}"
        )
    common = dict(steps=args.real_steps, size=int(cfg["size"]), two_views=False, seed=42)
    real_train = make_event_dataset("cifar10dvs", args.real_events_root, True, **common)
    real_test = make_event_dataset("cifar10dvs", args.real_events_root, False, **common)
    test_loader = DataLoader(
        real_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("metrics", {}).get("epoch"),
        "model_file": str(Path(model_module.__file__).resolve()),
        "student_training_domain": "CIFAR10-RCLS-medium synthetic events",
        "target_dataset": "real CIFAR10-DVS",
        "train_pool": len(real_train),
        "heldout": len(real_test),
        "shots": args.shots,
        "protocols": args.protocols,
        "representation": args.representation,
        "num_subsets": args.num_subsets,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "real_steps": args.real_steps,
        "augmentation": args.augmentation,
        "cutmix_probability": args.cutmix_probability,
        "warning": (
            "paper_comparable_best_heldout_accuracy selects on the held-out set, as in the "
            "SpikeCLR implementation; final_accuracy is also reported separately"
        ),
    }
    metrics_path = args.output_dir / "metrics.json"
    records: list[dict] = []
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        records = previous.get("records", [])
    completed = {(r["protocol"], int(r["shots"]), int(r["subset_index"])) for r in records}
    augment_fn = None
    if args.augmentation == "weak":
        augment_fn = lambda x: augment_events(x, "weak")

    for shot in args.shots:
        for subset_index in range(args.num_subsets):
            subset_seed = args.base_seed + subset_index * 1000
            indices = stratified_shot_indices(real_train, shot, subset_seed)
            for protocol in args.protocols:
                key = (protocol, shot, subset_index)
                if key in completed:
                    print(f"skip completed protocol={protocol} shots={shot} subset={subset_index}", flush=True)
                    continue
                run_seed = subset_seed + (0 if protocol == "lp" else 100_000)
                set_seed(run_seed)
                dataset = AugmentedSubset(real_train, indices, augment_fn=augment_fn)
                generator = torch.Generator().manual_seed(run_seed)
                train_loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    generator=generator,
                    num_workers=0,
                    pin_memory=True,
                )
                student = model_from_checkpoint(PureSpikeFormer, checkpoint, device)
                representation_dim = int(cfg["dim"])
                classifier = StudentClassifier(
                    student, representation_dim, classes=10, representation=args.representation,
                ).to(device)
                print(
                    f"start protocol={protocol} shots={shot} subset={subset_index} "
                    f"seed={subset_seed} samples={len(dataset)}",
                    flush=True,
                )
                result = train_one(
                    classifier=classifier,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    protocol=protocol,
                    epochs=args.epochs,
                    eval_every=args.eval_every,
                    lr=args.lr_linear if protocol == "lp" else args.lr_finetune,
                    weight_decay=args.weight_decay,
                    cutmix_probability=args.cutmix_probability,
                )
                result.update(
                    {
                        "protocol": protocol,
                        "shots": shot,
                        "subset_index": subset_index,
                        "subset_seed": subset_seed,
                        "sample_indices": indices,
                    }
                )
                records.append(result)
                payload = {
                    "metadata": metadata,
                    "aggregate": aggregate(records, args.shots, args.protocols),
                    "records": records,
                }
                metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                del classifier, student
                torch.cuda.empty_cache()
    print(json.dumps(aggregate(records, args.shots, args.protocols), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
