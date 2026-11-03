from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def flatten_row(row: dict) -> dict[str, object]:
    energy = row.get("energy") or {}
    return {
        "epoch": row.get("epoch"),
        "loss": row.get("loss"),
        "alignment": row.get("alignment"),
        "geometry": row.get("geometry"),
        "patch_affinity": row.get("patch_affinity"),
        "signed_knn": row.get("signed_knn"),
        "core_knn": row.get("core_knn"),
        "temporal_knn": row.get("temporal_knn"),
        "uniform_knn": row.get("uniform_knn"),
        "spike_rate": row.get("spike_rate"),
        "estimated_energy_pj": energy.get("estimated_energy_pj"),
        "estimated_energy_mj": (
            float(energy["estimated_energy_pj"]) / 1e9
            if energy.get("estimated_energy_pj") is not None
            else None
        ),
        "estimated_energy_ratio": energy.get("estimated_energy_ratio"),
        "activity_fraction": energy.get("activity_fraction"),
        "synops_per_sample": energy.get("synops_per_sample"),
        "labels_used": row.get("labels_used"),
        "supervised_loss_enabled": row.get("supervised_loss_enabled"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["epoch"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_dir: Path) -> dict[str, object]:
    history_path = run_dir / "history.json"
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    flat = [flatten_row(row) for row in rows]
    eval_rows = [row for row in flat if row.get("epoch") and int(row["epoch"]) % 5 == 0 and row.get("signed_knn") is not None]
    write_csv(run_dir / "training_trace.csv", flat)
    write_csv(run_dir / "eval_every5.csv", eval_rows)

    def best(metric: str) -> dict[str, object] | None:
        valid = [row for row in eval_rows if row.get(metric) is not None]
        return max(valid, key=lambda row: float(row[metric])) if valid else None

    latest = flat[-1] if flat else None
    manifest = {
        "run_dir": str(run_dir),
        "history_source": str(history_path),
        "checkpoint": str(run_dir / "best.pt"),
        "checkpoint_exists": (run_dir / "best.pt").exists(),
        "last_recorded_epoch": latest.get("epoch") if latest else None,
        "best_signed_knn": best("signed_knn"),
        "best_core_knn": best("core_knn"),
        "latest_evaluation": eval_rows[-1] if eval_rows else None,
        "loss_start": flat[0].get("loss") if flat else None,
        "loss_latest": latest.get("loss") if latest else None,
        "label_free_student_loss": all(row.get("labels_used") is False and row.get("supervised_loss_enabled") is False for row in flat),
        "eval_interval": 5,
    }
    (run_dir / "best_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        print(json.dumps(summarize(run_dir), indent=2))


if __name__ == "__main__":
    main()
