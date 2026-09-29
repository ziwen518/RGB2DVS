"""Evaluate a frozen RGB2DVS Student under downstream label budgets.

This script keeps label-free Student pretraining separate from downstream LP/FT.
For N-Caltech101 it honors the existing train/validation/test manifest; for
RCLS-style MotionEvents, the dataset's train/test directories are used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: list[int], augment: bool = False):
        self.base, self.indices, self.augment = base, list(indices), augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y, ordinal, sample_id = self.base[self.indices[index]]
        if self.augment:
            x = augment_events(x)
        return x, y, ordinal, sample_id


class HeadModel(nn.Module):
    def __init__(self, student: nn.Module, dim: int, classes: int, representation: str):
        super().__init__()
        self.student = student
        self.head = nn.Linear(dim, classes)
        self.representation = representation

    def forward(self, x):
        return self.head(self.student(x)[self.representation])


def augment_events(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    if random.random() < 0.5:
        y = y.flip(-1)
    if random.random() < 0.5:
        shift = max(1, y.shape[-1] // 12)
        y = y.roll(random.randint(-shift, shift), -1).roll(random.randint(-shift, shift), -2)
    if random.random() < 0.5:
        y = (y * random.uniform(0.9, 1.1)).clamp(0, 1)
    return y.contiguous()


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", choices=("cifar10rcls", "ncaltech101"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 9, 18, 45])
    p.add_argument(
        "--fractions",
        type=float,
        nargs="*",
        default=[],
        help="class-stratified label fractions in (0, 1), using one or more subset seeds",
    )
    p.add_argument("--include-full", action="store_true", help="also train on every target train sample")
    p.add_argument("--protocols", nargs="+", choices=("lp", "ft"), default=["lp", "ft"])
    p.add_argument("--num-subsets", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr-linear", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cutmix-probability", type=float, default=0.5)
    p.add_argument("--representation", choices=("embedding", "core_embedding"), default="embedding")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_model(model_module, checkpoint: dict, device: torch.device):
    c = checkpoint["args"]
    model = model_module.PureSpikeFormer(
        in_channels=2, dim=int(c["dim"]), depth=int(c["depth"]), heads=int(c["heads"]),
        patch_size=int(c["patch_size"]), threshold=float(c["threshold"]),
        use_cls_token=bool(c.get("use_cls_token", False)), norm="bntt", image_size=int(c["size"]),
        temporal_readout="learned", local_stem=not bool(c.get("pyramid_stem", False)),
        pyramid_stem=bool(c.get("pyramid_stem", False)), use_positional_bias=True,
        attention_mode="normalized", signed_readout=bool(c.get("signed_readout", True)),
        multidepth_readout=bool(c.get("multidepth_readout", False)),
        multidepth_readout_layers=int(c.get("multidepth_readout_layers", 3)),
        block_spiking=c.get("block_spiking", "full_respike"),
        hybrid_attention=bool(c.get("hybrid_attention", False)),
        hybrid_attention_suffix=int(c.get("hybrid_attention_suffix", 0)),
        qkv_temporal_mode=c.get("qkv_temporal_mode", "standard"),
        continuous=bool(c.get("continuous_relaxation", False)),
        ternary_threshold=float(c.get("ternary_threshold", 0.5)), temporal_steps=int(c["steps"]),
        population_bins=int(c.get("population_bins", 1)), event_prompt=bool(c.get("event_prompt", False)),
        prompt_strength=float(c.get("prompt_strength", 0.7)),
        local_structure_mixer=bool(c.get("local_structure_mixer", False)),
        local_mixer_dilations=tuple(c.get("local_mixer_dilations", [1, 2])),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def ids_for(base: Dataset) -> list[str]:
    return [str(s[0].relative_to(base.root)).replace("\\", "/") for s in base.samples]


def build_sets(a, data_module):
    if a.dataset == "ncaltech101":
        all_ds = data_module.make_event_dataset("ncaltech101", a.data_root, True, steps=a.steps, size=48, use_all=True)
        manifest = json.loads(a.split_manifest.read_text(encoding="utf-8-sig"))
        lookup = {sample_id: i for i, sample_id in enumerate(ids_for(all_ds))}
        def select(key):
            return IndexedDataset(all_ds, [lookup[x] for x in manifest[f"{key}_ids"]])
        return select("train"), select("validation"), select("test"), len(manifest["classes"])
    train = data_module.make_event_dataset("motion", a.data_root, True, steps=a.steps, size=48, frame_mode="count_global")
    test = data_module.make_event_dataset("motion", a.data_root, False, steps=a.steps, size=48, frame_mode="count_global")
    classes = len(train.class_names)
    labels = np.asarray([int(train[i][1]) for i in range(len(train))])
    rng = np.random.default_rng(12345)
    fit, valid = [], []
    for c in range(classes):
        idx = np.flatnonzero(labels == c)
        n_valid = min(max(1, int(round(len(idx) * 0.1))), len(idx) - 1)
        rng.shuffle(idx)
        valid.extend(idx[:n_valid].tolist())
        fit.extend(idx[n_valid:].tolist())
    rng.shuffle(fit); rng.shuffle(valid)
    return IndexedDataset(train, fit), IndexedDataset(train, valid), test, classes


def stratified_indices(ds: Dataset, shots: int, seed: int, classes: int):
    labels = np.asarray([int(ds[i][1]) for i in range(len(ds))])
    rng = np.random.default_rng(seed); selected = []
    for c in range(classes):
        idx = np.flatnonzero(labels == c)
        if len(idx) < shots: raise ValueError(f"class {c} has {len(idx)} samples, need {shots}")
        selected.extend(rng.choice(idx, shots, replace=False).tolist())
    rng.shuffle(selected); return selected


def stratified_fraction_indices(ds: Dataset, fraction: float, seed: int, classes: int):
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    labels = np.asarray([int(ds[i][1]) for i in range(len(ds))])
    rng = np.random.default_rng(seed); selected = []
    for c in range(classes):
        idx = np.flatnonzero(labels == c)
        count = max(1, int(len(idx) * fraction))
        selected.extend(rng.choice(idx, count, replace=False).tolist())
    rng.shuffle(selected); return selected


def all_indices(ds: Dataset):
    return list(range(len(ds)))


@torch.inference_mode()
def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    for x, y, *_ in loader:
        logits = model(x.to(device, non_blocking=True))
        y = torch.as_tensor(y, device=device)
        correct += int(logits.argmax(-1).eq(y).sum()); total += int(y.numel())
    return correct / max(1, total)


def run_one(student, train, valid, test, classes, a, protocol, shot, subset, seed, dim, device, chosen=None):
    seed_all(seed)
    chosen = stratified_indices(train, shot, seed, classes) if chosen is None else chosen
    train_ds = IndexedDataset(train, chosen, augment=True)
    tr = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0)
    va = DataLoader(valid, a.eval_batch_size, shuffle=False, num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0)
    te = DataLoader(test, a.eval_batch_size, shuffle=False, num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0)
    model = HeadModel(student, dim, classes, a.representation).to(device)
    freeze = protocol == "lp"
    for p in model.student.parameters(): p.requires_grad_(not freeze)
    params = model.head.parameters() if freeze else model.parameters()
    opt = torch.optim.AdamW(params, lr=a.lr_linear if freeze else a.lr_finetune, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    best, best_ep = -1.0, -1; best_state = None; history=[]; start=time.perf_counter()
    for ep in range(1, a.epochs + 1):
        model.train()
        if freeze: model.student.eval()
        loss_sum = 0.0; count = 0
        for x, y, *_ in tr:
            x=x.to(device, non_blocking=True); y=torch.as_tensor(y, device=device)
            opt.zero_grad(set_to_none=True); logits=model(x); loss=F.cross_entropy(logits,y); loss.backward(); opt.step()
            loss_sum += float(loss.detach()) * y.numel(); count += int(y.numel())
        sched.step(); row={"epoch":ep,"loss":loss_sum/max(1,count)}
        if ep == a.epochs or ep % a.eval_every == 0:
            row["validation_accuracy"] = accuracy(model, va, device)
            if row["validation_accuracy"] > best:
                best=row["validation_accuracy"]
                best_ep=ep
                # Test only the checkpoint selected by validation, never the final epoch.
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    final = accuracy(model, te, device)
    return {"protocol":protocol,"shots":shot,"subset_index":subset,"seed":seed,"selected":chosen,
            "best_validation_accuracy":best,"best_validation_epoch":best_ep,"final_test_accuracy":final,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "elapsed_seconds":time.perf_counter()-start,"history":history}


def main():
    a=args(); a.output_dir.mkdir(parents=True, exist_ok=True); sys.path.insert(0,str(a.project_root/"code"/"core"))
    import model as model_module
    from data import make_event_dataset
    device=torch.device("cuda",a.gpu); torch.cuda.set_device(a.gpu)
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if int(ckpt["args"]["steps"]) != a.steps: raise ValueError("checkpoint/data steps mismatch")
    train, valid, test, classes=build_sets(a, type("D",(),{"make_event_dataset":staticmethod(make_event_dataset)})())
    dim=int(ckpt["args"]["dim"]); records=[]; path=a.output_dir/"metrics.json"
    if path.exists(): records=json.loads(path.read_text(encoding="utf-8")).get("records",[])
    done={(r["protocol"],str(r["budget"]),int(r["subset_index"])) for r in records}
    metadata={"checkpoint":str(a.checkpoint),"dataset":a.dataset,"train_samples":len(train),"validation_samples":len(valid),"test_samples":len(test),"classes":classes,"model_file":str(Path(model_module.__file__).resolve()),"representation":a.representation,"shots":a.shots,"fractions":a.fractions,"protocols":a.protocols,"num_subsets":a.num_subsets,"epochs":a.epochs,"labels_used_only_downstream":True,"student_labels_used":False}
    budgets = [(str(s), s, None) for s in a.shots]
    budgets.extend((f"fraction_{fraction:g}", None, fraction) for fraction in a.fractions)
    if a.include_full:
        budgets.append(("full", None, None))
    for budget, shot, fraction in budgets:
        for subset in range(1 if budget == "full" else a.num_subsets):
            for protocol in a.protocols:
                key=(protocol,budget,subset)
                if key in done: continue
                seed=a.base_seed+subset*1000+(100000 if protocol=="ft" else 0)
                student=load_model(model_module,ckpt,device)
                if budget == "full":
                    chosen = all_indices(train)
                elif fraction is not None:
                    chosen = stratified_fraction_indices(train, fraction, seed, classes)
                else:
                    chosen = None
                rec=run_one(student,train,valid,test,classes,a,protocol,shot or 0,subset,seed,dim,device,chosen)
                rec["budget"] = budget
                if fraction is not None:
                    rec["fraction"] = fraction
                records.append(rec); del student; torch.cuda.empty_cache()
                payload={"metadata":metadata,"records":records}
                path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
                print(json.dumps({k:rec[k] for k in ("protocol","shots","subset_index","best_validation_accuracy","final_test_accuracy")}),flush=True)
    print(json.dumps({"records":len(records),"output":str(path)},indent=2),flush=True)


if __name__ == "__main__": main()
