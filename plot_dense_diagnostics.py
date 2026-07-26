"""Plot dense degradation diagnostics from saved patch/VOC summaries."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dense_results_io import read_csv_rows, read_voc_results


VOC_V2_METRIC_VERSION = "global_confusion_v2"


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


def read_protocol_voc_results(
    path: str | Path | None,
    *,
    protocol: str,
    metric_version: str | None,
    probe_seed: int | None,
    checkpoint_key: str | None,
) -> tuple[dict[int, float], dict[str, str | int]]:
    """Read validated v2 results or explicitly selected historical results."""
    if path is None:
        return {}, {
            "metric_version": "",
            "probe_seed": "",
            "checkpoint_key": "",
        }
    if protocol == "legacy":
        if any(value is not None for value in (metric_version, probe_seed, checkpoint_key)):
            raise ValueError(
                "Legacy VOC mode does not accept v2 protocol expectations"
            )
        rows = read_voc_results(path, as_rows=True)
        unexpected_versions = {
            row.get("metric_version")
            for row in rows
            if row.get("metric_version") not in (None, "batch_mean_v1")
        }
        if unexpected_versions:
            raise ValueError(
                "Legacy VOC mode accepts only unversioned or batch_mean_v1 rows; "
                f"got {sorted(unexpected_versions)}"
            )
        return {int(row["epoch"]): float(row["miou"]) for row in rows}, {
            "metric_version": "batch_mean_v1",
            "probe_seed": "unrecorded",
            "checkpoint_key": "unrecorded",
        }
    if protocol != "v2":
        raise ValueError(f"Unsupported VOC protocol: {protocol}")
    if metric_version is None or probe_seed is None or checkpoint_key is None:
        raise ValueError(
            "VOC v2 mode requires --voc_metric_version, --voc_probe_seed, "
            "and --voc_checkpoint_key"
        )
    if metric_version != VOC_V2_METRIC_VERSION:
        raise ValueError(
            f"VOC v2 mode requires metric version {VOC_V2_METRIC_VERSION}; "
            f"got {metric_version}"
        )
    return read_voc_results(
        path,
        expected_metric_version=metric_version,
        expected_probe_seed=probe_seed,
        expected_checkpoint_key=checkpoint_key,
        require_provenance=True,
    ), {
        "metric_version": metric_version,
        "probe_seed": probe_seed,
        "checkpoint_key": checkpoint_key,
    }


def write_combined_summary(
    path: Path,
    rows: list[dict],
    voc_by_epoch: dict[int, float],
    *,
    voc_metric_version: str | None = None,
    voc_probe_seed: int | str | None = None,
    voc_checkpoint_key: str | None = None,
) -> None:
    if not rows:
        return
    if voc_by_epoch and any(
        value is None
        for value in (voc_metric_version, voc_probe_seed, voc_checkpoint_key)
    ):
        raise ValueError("VOC protocol metadata is required for a combined summary")
    keys = [
        "epoch",
        "voc_miou",
        "voc_metric_version",
        "voc_probe_seed",
        "voc_checkpoint_key",
    ] + [key for key in rows[0].keys() if key != "epoch"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            combined = {
                "epoch": row["epoch"],
                "voc_miou": voc_by_epoch.get(int(row["epoch"]), ""),
                "voc_metric_version": voc_metric_version or "",
                "voc_probe_seed": (
                    voc_probe_seed if voc_probe_seed is not None else ""
                ),
                "voc_checkpoint_key": voc_checkpoint_key or "",
            }
            combined.update({key: row.get(key, "") for key in keys if key not in combined})
            writer.writerow(combined)


def plot_raw_l2_figures(rows: list[dict], out_dir: str | Path) -> list[Path]:
    """Write focused raw-vs-L2 diagnostic figures when both tracks exist."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    specs = [
        (
            "fig_raw_vs_l2_dse.png",
            "Raw vs L2 DSE",
            [("raw_dse", "raw DSE"), ("l2_dse", "L2 DSE")],
            "DSE",
        ),
        (
            "fig_raw_vs_l2_class_sep.png",
            "Raw vs L2 class separability",
            [("raw_class_sep_avg", "raw class sep"), ("l2_class_sep_avg", "L2 class sep")],
            "M_inter - M_intra",
        ),
    ]

    for filename, title, metrics, ylabel in specs:
        fig, axis = plt.subplots(figsize=(7, 4))
        any_metric = False
        for metric, label in metrics:
            any_metric = plot_metric(axis, rows, metric, label) or any_metric
        if not any_metric:
            plt.close(fig)
            continue
        axis.set_title(title)
        axis.set_xlabel("checkpoint epoch")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    any_left = False
    for metric, label in [
        ("raw_effective_rank", "raw effective rank"),
        ("l2_effective_rank", "L2 effective rank"),
    ]:
        any_left = plot_metric(axes[0], rows, metric, label) or any_left
    axes[0].set_title("Effective rank")
    axes[0].set_xlabel("checkpoint epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    any_right = False
    for metric, label in [
        ("raw_top1_eigen_ratio", "raw top-1 ratio"),
        ("l2_top1_eigen_ratio", "L2 top-1 ratio"),
    ]:
        any_right = plot_metric(axes[1], rows, metric, label) or any_right
    axes[1].set_title("Spectrum concentration")
    axes[1].set_xlabel("checkpoint epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    if any_left or any_right:
        path = out_dir / "fig_raw_vs_l2_spectrum.png"
        fig.savefig(path, dpi=180)
        written.append(path)
    plt.close(fig)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--voc_json", default=None)
    parser.add_argument("--voc_protocol", choices=["v2", "legacy"], default="v2")
    parser.add_argument("--voc_metric_version", default=None)
    parser.add_argument("--voc_probe_seed", type=int, default=None)
    parser.add_argument(
        "--voc_checkpoint_key",
        choices=["teacher", "student"],
        default=None,
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    rows = read_csv_rows(args.summary_csv)
    voc_by_epoch, voc_metadata = read_protocol_voc_results(
        args.voc_json,
        protocol=args.voc_protocol,
        metric_version=args.voc_metric_version,
        probe_seed=args.voc_probe_seed,
        checkpoint_key=args.voc_checkpoint_key,
    )
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
    if voc_by_epoch:
        protocol_label = (
            f"{voc_metadata['metric_version']}, seed={voc_metadata['probe_seed']}, "
            f"key={voc_metadata['checkpoint_key']}"
        )
        axes[0].set_title(f"PASCAL VOC linear mIoU\n{protocol_label}")
    else:
        axes[0].set_title("PASCAL VOC linear mIoU")
    axes[0].set_xlabel("checkpoint epoch")
    axes[0].grid(alpha=0.3)

    for metric, label in [
        ("dse", "DSE"),
        ("class_sep_avg", "class separability"),
        ("effective_rank", "effective rank"),
    ]:
        plot_metric(axes[1], rows, metric, label)
    axes[1].set_title("DSE structural metrics")
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

    for path in plot_raw_l2_figures(rows, out_dir):
        print(f"saved raw/L2 figure: {path}")
    write_combined_summary(
        out_dir / "combined_dense_summary.csv",
        rows,
        voc_by_epoch,
        voc_metric_version=voc_metadata["metric_version"] or None,
        voc_probe_seed=voc_metadata["probe_seed"],
        voc_checkpoint_key=voc_metadata["checkpoint_key"] or None,
    )
    print(f"saved figure: {fig_path}")
    print(f"saved combined summary: {out_dir / 'combined_dense_summary.csv'}")


if __name__ == "__main__":
    main()
