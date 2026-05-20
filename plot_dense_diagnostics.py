"""Plot dense degradation diagnostics from saved patch/VOC summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_number(value):
    if value is None or value == "":
        return float("nan")
    try:
        if isinstance(value, str) and value.strip().isdigit():
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return value


def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        rows = [{key: to_number(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return sorted(rows, key=lambda row: row["epoch"])


def read_voc_results(path: str | Path | None) -> dict[int, float]:
    if not path or not Path(path).is_file():
        return {}
    with Path(path).open() as handle:
        payload = json.load(handle)
    return {int(row["epoch"]): float(row["miou"]) for row in payload}


def metric_values(rows: list[dict], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(metric, float("nan"))
        values.append(float(value) if isinstance(value, (int, float)) else float("nan"))
    return values


def plot_metric(axis, rows: list[dict], metric: str, label: str | None = None, **kwargs) -> bool:
    values = metric_values(rows, metric)
    if all(math.isnan(value) for value in values):
        return False
    axis.plot([row["epoch"] for row in rows], values, marker="o", label=label or metric, **kwargs)
    return True


def write_combined_summary(path: Path, rows: list[dict], voc_by_epoch: dict[int, float]) -> None:
    if not rows:
        return
    keys = ["epoch", "voc_miou"] + [key for key in rows[0].keys() if key != "epoch"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            combined = {"epoch": row["epoch"], "voc_miou": voc_by_epoch.get(int(row["epoch"]), "")}
            combined.update({key: row.get(key, "") for key in keys if key not in combined})
            writer.writerow(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--voc_json", default=None)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows = read_csv_rows(args.summary_csv)
    voc_by_epoch = read_voc_results(args.voc_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.reshape(-1)
    epochs = [row["epoch"] for row in rows]

    if voc_by_epoch:
        voc_values = [voc_by_epoch.get(int(epoch), float("nan")) for epoch in epochs]
        axes[0].plot(epochs, voc_values, marker="o", color="tab:blue", label="VOC mIoU")
        best_epoch = max(voc_by_epoch, key=voc_by_epoch.get)
        axes[0].axvline(best_epoch, color="tab:red", linestyle="--", alpha=0.5, label=f"best {best_epoch}")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "VOC results not provided", ha="center", va="center")
    axes[0].set_title("PASCAL VOC linear mIoU")
    axes[0].set_xlabel("checkpoint epoch")
    axes[0].grid(alpha=0.3)

    for metric, label in [
        ("dse", "DSE"),
        ("class_sep_avg", "class separability"),
        ("effective_rank", "effective rank"),
    ]:
        plot_metric(axes[1], rows, metric, label)
    axes[1].set_title("DSE-style structural metrics")
    axes[1].set_xlabel("checkpoint epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plot_metric(axes[2], rows, "effective_rank", "singular effective rank", color="tab:green")
    right = axes[2].twinx()
    plot_metric(right, rows, "top1_eigen_ratio", "top-1 eigen ratio", color="tab:red")
    axes[2].set_title("Feature dimensionality and spectrum concentration")
    axes[2].set_xlabel("checkpoint epoch")
    axes[2].grid(alpha=0.3)

    for metric, label in [
        ("cls_patch_cos_mean", "mean"),
        ("cls_patch_cos_p90", "p90"),
    ]:
        plot_metric(axes[3], rows, metric, label)
    axes[3].set_title("CLS-patch cosine similarity")
    axes[3].set_xlabel("checkpoint epoch")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    for metric, label in [
        ("cls_attention_entropy_mean", "attention entropy"),
        ("cls_attention_top10_mass_mean", "top-10 mass"),
    ]:
        plot_metric(axes[4], rows, metric, label)
    axes[4].set_title("CLS attention concentration")
    axes[4].set_xlabel("checkpoint epoch")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    for metric, label in [
        ("patch_norm_mean", "mean"),
        ("patch_norm_p90", "p90"),
        ("query_sim_entropy_mean", "query entropy"),
    ]:
        plot_metric(axes[5], rows, metric, label)
    axes[5].set_title("Patch magnitude and query-map entropy")
    axes[5].set_xlabel("checkpoint epoch")
    axes[5].legend()
    axes[5].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "fig_dense_diagnostics_summary.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    write_combined_summary(out_dir / "combined_dense_summary.csv", rows, voc_by_epoch)
    print(f"saved figure: {fig_path}")
    print(f"saved combined summary: {out_dir / 'combined_dense_summary.csv'}")


if __name__ == "__main__":
    main()
