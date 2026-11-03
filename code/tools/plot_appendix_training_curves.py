"""Plot label-free Student training curves from formal history files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar-history", type=Path, required=True)
    parser.add_argument("--ncaltech-history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    histories = [
        ("CIFAR10-RCLS", json.loads(args.cifar_history.read_text(encoding="utf-8-sig"))),
        ("N-Caltech101", json.loads(args.ncaltech_history.read_text(encoding="utf-8-sig"))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), dpi=220, sharex="col")
    summary = {}
    for column, (name, rows) in enumerate(histories):
        epochs = [row["epoch"] for row in rows]
        loss = [row["loss"] for row in rows]
        axes[0, column].plot(epochs, loss, color="#2a6fbb", linewidth=1.5)
        axes[0, column].set_title(name, fontsize=10)
        axes[0, column].set_ylabel("Distillation loss" if column == 0 else "")
        eval_rows = [row for row in rows if row.get("signed_knn") is not None]
        eval_epochs = [row["epoch"] for row in eval_rows]
        signed = [100.0 * row["signed_knn"] for row in eval_rows]
        core = [100.0 * row["core_knn"] for row in eval_rows]
        axes[1, column].plot(eval_epochs, signed, color="#c43b3b", marker="o", markersize=2.8,
                             linewidth=1.4, label="Signed readout")
        axes[1, column].plot(eval_epochs, core, color="#27966a", marker="s", markersize=2.5,
                             linewidth=1.3, label="Core")
        axes[1, column].set_xlabel("Epoch")
        axes[1, column].set_ylabel("Validation kNN@10 (%)" if column == 0 else "")
        axes[1, column].legend(frameon=False, fontsize=7, loc="lower right")
        for axis in (axes[0, column], axes[1, column]):
            axis.grid(alpha=0.22, linewidth=0.6)
            axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
        summary[name] = {
            "best_signed_knn": max(signed), "best_signed_epoch": eval_epochs[signed.index(max(signed))],
            "best_core_knn": max(core), "best_core_epoch": eval_epochs[core.index(max(core))],
            "final_loss": loss[-1],
        }
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "student_training_curves.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "student_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    (args.output_dir / "training_curve_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
