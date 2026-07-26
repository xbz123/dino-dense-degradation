"""Create a Markdown report for dense degradation diagnostic runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from dense_results_io import read_csv_rows, read_voc_results


VOC_V2_METRIC_VERSION = "global_confusion_v2"


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def delta(first: dict, last: dict, metric: str) -> str:
    a = first.get(metric, float("nan"))
    b = last.get(metric, float("nan"))
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or math.isnan(a) or math.isnan(b):
        return "n/a"
    return f"{fmt(a)} -> {fmt(b)} ({b - a:+.4f})"


def read_protocol_voc_rows(
    path: str | Path | None,
    *,
    protocol: str,
    metric_version: str | None,
    probe_seed: int | None,
    checkpoint_key: str | None,
) -> list[dict]:
    """Read either validated v2 rows or explicitly requested historical rows."""
    if path is None:
        return []
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
        return rows
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
        as_rows=True,
        expected_metric_version=metric_version,
        expected_probe_seed=probe_seed,
        expected_checkpoint_key=checkpoint_key,
        require_provenance=True,
    )


def voc_section(voc_rows: list[dict], *, protocol: str) -> list[str]:
    if not voc_rows:
        return [
            "## Downstream VOC",
            "",
            "VOC mIoU results were not provided for this report.",
        ]
    best = max(voc_rows, key=lambda row: row["miou"])
    first = voc_rows[0]
    last = voc_rows[-1]
    drop_from_best = last["miou"] - best["miou"]
    if drop_from_best < -1.0:
        interpretation = "The final checkpoint is clearly below the best VOC checkpoint."
    elif drop_from_best < -0.2:
        interpretation = "The final checkpoint is slightly below the best VOC checkpoint."
    else:
        interpretation = "The final checkpoint remains close to the best VOC checkpoint; VOC alone does not show a clear downstream degradation drop."
    if protocol == "legacy":
        interpretation = (
            "Historical batch-mean-v1 evidence only; this curve is not eligible "
            "for metric-v2 phenomenon confirmation or a v10 decision gate. "
            + interpretation
        )
    return [
        "## Downstream VOC",
        "",
        f"- Evaluated checkpoints: {len(voc_rows)}",
        f"- First checkpoint: epoch {int(first['epoch'])}, mIoU {fmt(first['miou'], 3)}",
        f"- Best checkpoint: epoch {int(best['epoch'])}, mIoU {fmt(best['miou'], 3)}",
        f"- Final checkpoint: epoch {int(last['epoch'])}, mIoU {fmt(last['miou'], 3)}",
        f"- Final minus best: {drop_from_best:+.3f}",
        "",
        interpretation,
    ]


def voc_provenance_lines(voc_rows: list[dict], *, protocol: str) -> list[str]:
    if not voc_rows:
        return ["- VOC provenance: not applicable"]
    if protocol == "legacy":
        return ["- VOC provenance: unavailable for historical batch-mean-v1 rows"]
    first = voc_rows[0]
    return [
        f"- VOC representation: `{first['representation']}`",
        f"- VOC source commit: `{first['source_commit']}`",
        f"- VOC source dirty: `{first['source_dirty']}`",
        f"- VOC dataset: `{first['dataset_identity'].get('name', 'unrecorded')}`",
        f"- VOC checkpoint hashes recorded: `{len(voc_rows)}`",
    ]


def diagnostics_section(rows: list[dict]) -> list[str]:
    if not rows:
        return ["## Structural Diagnostics", "", "No patch diagnostic rows were found."]
    first = rows[0]
    last = rows[-1]
    lines = [
        "## Structural Diagnostics",
        "",
        f"- Checkpoint range: epoch {int(first['epoch'])} to epoch {int(last['epoch'])}",
        f"- DSE: {delta(first, last, 'dse')}",
        f"- Class separability: {delta(first, last, 'class_sep_avg')}",
        f"- Effective rank: {delta(first, last, 'effective_rank')}",
        f"- Top-1 eigenvalue ratio: {delta(first, last, 'top1_eigen_ratio')}",
        f"- CLS-patch cosine mean: {delta(first, last, 'cls_patch_cos_mean')}",
        f"- Patch norm mean: {delta(first, last, 'patch_norm_mean')}",
        f"- Query similarity entropy: {delta(first, last, 'query_sim_entropy_mean')}",
    ]
    if "raw_dse" in first and "l2_dse" in first:
        lines.extend(
            [
                "",
                "### Raw vs L2 Structural Tracks",
                "",
                f"- Raw DSE: {delta(first, last, 'raw_dse')}",
                f"- L2 DSE: {delta(first, last, 'l2_dse')}",
                f"- Raw class separability: {delta(first, last, 'raw_class_sep_avg')}",
                f"- L2 class separability: {delta(first, last, 'l2_class_sep_avg')}",
                f"- Raw effective rank: {delta(first, last, 'raw_effective_rank')}",
                f"- L2 effective rank: {delta(first, last, 'l2_effective_rank')}",
                f"- Raw top-1 eigenvalue ratio: {delta(first, last, 'raw_top1_eigen_ratio')}",
                f"- L2 top-1 eigenvalue ratio: {delta(first, last, 'l2_top1_eigen_ratio')}",
            ]
        )
    return lines


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
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_csv_rows(args.summary_csv)
    voc_rows = read_protocol_voc_rows(
        args.voc_json,
        protocol=args.voc_protocol,
        metric_version=args.voc_metric_version,
        probe_seed=args.voc_probe_seed,
        checkpoint_key=args.voc_checkpoint_key,
    )

    lines = [
        "# Dense Degradation Diagnostics Report",
        "",
        "## Setup",
        "",
        f"- Patch diagnostics: `{args.summary_csv}`",
        f"- VOC results: `{args.voc_json}`" if args.voc_json else "- VOC results: not provided",
        f"- VOC protocol: `{args.voc_protocol}`",
        (
            f"- VOC metric version: `{args.voc_metric_version}`"
            if args.voc_json and args.voc_protocol == "v2"
            else "- VOC metric version: `batch_mean_v1` (historical only)"
            if args.voc_json
            else "- VOC metric version: not applicable"
        ),
        (
            f"- VOC probe seed: `{args.voc_probe_seed}`"
            if args.voc_json and args.voc_protocol == "v2"
            else "- VOC probe seed: unrecorded"
            if args.voc_json
            else "- VOC probe seed: not applicable"
        ),
        (
            f"- VOC checkpoint key: `{args.voc_checkpoint_key}`"
            if args.voc_json and args.voc_protocol == "v2"
            else "- VOC checkpoint key: unrecorded"
            if args.voc_json
            else "- VOC checkpoint key: not applicable"
        ),
        *voc_provenance_lines(voc_rows, protocol=args.voc_protocol),
        f"- Number of patch diagnostic checkpoints: {len(rows)}",
        "",
        *voc_section(voc_rows, protocol=args.voc_protocol),
        "",
        *diagnostics_section(rows),
        "",
        "## Interpretation",
        "",
        "Use this report as a compact checkpoint summary. VOC mIoU is downstream evidence. Raw DSE, class separability, and covariance metrics are Euclidean/covariance diagnostics on final-LayerNorm patch tokens and can be confounded by patch feature norm drift. L2-normalized tracks are the stronger check for angular patch-geometry degradation. CLS-patch cosine and fixed-query similarity are already L2-normalized diagnostics.",
        "",
        "## Key Outputs",
        "",
        "- `patch_attention_dse_summary.csv`: per-checkpoint structural metrics.",
        "- `combined_dense_summary.csv`: patch diagnostics merged with VOC mIoU when available.",
        "- `fig_dense_diagnostics_summary.png`: summary curves.",
        "- `fig_raw_vs_l2_dse.png`, `fig_raw_vs_l2_class_sep.png`, and `fig_raw_vs_l2_spectrum.png`: raw/L2 validation curves.",
        "- `epoch_XXXX/`: fixed-image PCA maps, CLS attention maps, patch norm maps, CLS similarity maps, query similarity maps, and histograms.",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"saved report: {out}")


if __name__ == "__main__":
    main()
