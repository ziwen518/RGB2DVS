"""Stream-extract N-ImageNet-mini event/RGB pairs from the downloaded archives."""
from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path


def safe_name(name: str) -> str:
    return Path(name).name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--validation-per-class", type=int, default=100)
    p.add_argument("--test-per-class", type=int, default=100)
    args = p.parse_args()
    root, out = args.root, args.output
    event_root = root / "event_part1" / "Part_1"
    rgb_zip = root / "training" / "Part_1.zip"
    event_out, rgb_out = out / "events", out / "rgb"
    event_out.mkdir(parents=True, exist_ok=True)
    rgb_out.mkdir(parents=True, exist_ok=True)
    classes = sorted(p.name.removesuffix(".tar.gz") for p in event_root.glob("*.tar.gz"))
    records = []
    missing_rgb, missing_event = [], []
    with zipfile.ZipFile(rgb_zip, "r") as outer:
        names = set(outer.namelist())
        for class_id, synset in enumerate(classes):
            event_tar = event_root / f"{synset}.tar.gz"
            rgb_member = f"Part_1/{synset}.tar.gz"
            if rgb_member not in names:
                missing_rgb.append(synset)
                continue
            rgb_by_stem = {}
            with outer.open(rgb_member) as stream, tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    stem = Path(member.name).stem
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    destination = rgb_out / synset / f"{stem}.JPEG"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                    rgb_by_stem[stem] = str(destination.relative_to(out)).replace("\\", "/")
            with tarfile.open(event_tar, mode="r:gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.lower().endswith(".npz"):
                        continue
                    stem = Path(member.name).stem
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    destination = event_out / synset / f"{stem}.npz"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                    event_rel = str(destination.relative_to(out)).replace("\\", "/")
                    rgb_rel = rgb_by_stem.get(stem)
                    if rgb_rel is None:
                        missing_rgb.append(f"{synset}/{stem}")
                        continue
                    records.append({"id": f"{synset}/{stem}", "class_name": synset,
                                    "class_id": class_id, "event_path": event_rel,
                                    "rgb_path": rgb_rel})
            print(json.dumps({"class": synset, "pairs_so_far": len(records)}), flush=True)
    # Deterministic balanced split; labels are not used in Student optimization.
    import random
    rng = random.Random(args.seed)
    by_class = {syn: [r for r in records if r["class_name"] == syn] for syn in classes}
    split = {"train": [], "validation": [], "test": [], "seed": args.seed, "classes": classes}
    for syn in classes:
        group = list(by_class.get(syn, [])); rng.shuffle(group)
        n_test = min(args.test_per_class, len(group) // 3)
        n_val = min(args.validation_per_class, max(0, (len(group) - n_test) // 3))
        split["test"].extend(r["event_path"] for r in group[:n_test])
        split["validation"].extend(r["event_path"] for r in group[n_test:n_test + n_val])
        split["train"].extend(r["event_path"] for r in group[n_test + n_val:])
    manifest = out / "manifests"; manifest.mkdir(parents=True, exist_ok=True)
    (manifest / "paired_all.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (manifest / "event_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    report = {"classes": len(classes), "pairs": len(records), "missing_rgb": missing_rgb,
              "missing_event": missing_event, "train": len(split["train"]),
              "validation": len(split["validation"]), "test": len(split["test"])}
    (manifest / "pairing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
