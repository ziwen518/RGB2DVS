from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path


URLS = {
    "images.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
    "lists.tar": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
}


def download_resume(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except Exception:
        if existing:
            destination.unlink()
            return download_resume(url, destination)
        raise
    status = getattr(response, "status", None)
    if existing and status != 206:
        destination.unlink()
        return download_resume(url, destination)
    mode = "ab" if existing and status == 206 else "wb"
    total = existing + int(response.headers.get("Content-Length", "0"))
    written = existing
    with destination.open(mode) as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if total:
                print(json.dumps({"file": destination.name, "bytes": written, "total": total}), flush=True)


def extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / f".{archive.stem}.extracted"
    if marker.exists():
        return
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(destination)
    marker.touch()


def read_mat(path: Path) -> list[str]:
    from scipy.io import loadmat

    payload = loadmat(path)
    if "file_list" not in payload:
        raise RuntimeError(f"official list has no file_list field: {path}")
    values = payload["file_list"]
    result: list[str] = []
    for value in values.reshape(-1):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result


def make_split_tree(raw_root: Path, output_root: Path) -> dict:
    images_root = raw_root / "Images"
    lists_root = raw_root / "lists"
    train = read_mat(lists_root / "train_list.mat")
    test = read_mat(lists_root / "test_list.mat")
    if len(train) != 12000 or len(test) != 8580:
        raise RuntimeError(f"unexpected official split sizes: train={len(train)} test={len(test)}")

    if output_root.exists():
        shutil.rmtree(output_root)
    for split, names in (("train", train), ("test", test)):
        for relative_name in names:
            source = images_root / relative_name
            class_name = Path(relative_name).parent.name
            target = output_root / split / class_name / Path(relative_name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    labels = sorted({Path(name).parent.name for name in train + test})
    report = {
        "source": "Stanford Dogs / ImageNetDogs official 120-class split",
        "classes": len(labels),
        "train": len(train),
        "test": len(test),
        "labels": labels,
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archives = args.staging / "archives"
    raw = args.staging / "raw"
    for name, url in URLS.items():
        download_resume(url, archives / name)
    extract_tar(archives / "lists.tar", raw)
    extract_tar(archives / "images.tar", raw)
    report = make_split_tree(raw, args.output)
    print(json.dumps({"completed": True, **report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
