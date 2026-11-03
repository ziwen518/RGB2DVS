from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

from scipy.io import loadmat


def read_list(path: Path) -> list[str]:
    values = loadmat(path)["file_list"].reshape(-1)
    result = []
    for value in values:
        value = value.item() if hasattr(value, "item") else value
        result.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.stage / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lists_archive = args.stage / "archives" / "lists.tar"
    images_archive = args.stage / "images.tar"
    with tarfile.open(lists_archive, "r:*") as archive:
        archive.extractall(raw)
    with tarfile.open(images_archive, "r:*") as archive:
        archive.extractall(raw)
    images_root = raw / "Images"
    train = read_list(raw / "train_list.mat")
    test = read_list(raw / "test_list.mat")
    classes = sorted(path.name for path in images_root.iterdir() if path.is_dir())
    if len(train) != 12000 or len(test) != 8580 or len(classes) != 120:
        raise RuntimeError(f"official count failed train={len(train)} test={len(test)} classes={len(classes)}")
    if args.output.exists():
        shutil.rmtree(args.output)
    for split, names in (("train", train), ("test", test)):
        for relative in names:
            source = images_root / relative
            target = args.output / split / Path(relative).parent.name / Path(relative).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    report = {
        "source": "Stanford Dogs official 120-class split",
        "classes": len(classes),
        "train": len(train),
        "test": len(test),
        "class_names": classes,
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"official_count_check": "passed", **report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
