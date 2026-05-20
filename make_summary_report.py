"""Create a Markdown report for dense degradation diagnostic runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


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


def read_voc_results(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    with Path(path).open() as handle:
        return sorted(json.load(handle), key=lambda row: row["epoch"])


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


def voc_section(voc_rows: list[dict]) -> list[str]:
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


def diagnostics_section(rows: list[dict]) -> list[str]:
    if not rows:
        return ["## Structural Diagnostics", "", "No patch diagnostic rows were found."]
    first = rows[0]
    last = rows[-1]
    return [
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--voc_json", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_csv_rows(args.summary_csv)
    voc_rows = read_voc_results(args.voc_json)

    lines = [
        "# Dense Degradation Diagnostics Report",
        "",
        "## Setup",
        "",
        f"- Patch diagnostics: `{args.summary_csv}`",
        f"- VOC results: `{args.voc_json}`" if args.voc_json else "- VOC results: not provided",
        f"- Number of patch diagnostic checkpoints: {len(rows)}",
        "",
        *voc_section(voc_rows),
        "",
        *diagnostics_section(rows),
        "",
        "## Interpretation",
        "",
        "Use this report as a compact checkpoint summary. VOC mIoU is downstream evidence, while DSE, effective rank, CLS-patch cosine, attention concentration, and fixed-query similarity are structural diagnostics. If VOC is stable but structural metrics drift, the current setup may show representation changes before downstream degradation becomes visible.",
        "",
        "## Key Outputs",
        "",
        "- `patch_attention_dse_summary.csv`: per-checkpoint structural metrics.",
        "- `combined_dense_summary.csv`: patch diagnostics merged with VOC mIoU when available.",
        "- `fig_dense_diagnostics_summary.png`: summary curves.",
        "- `epoch_XXXX/`: fixed-image PCA maps, CLS attention maps, patch norm maps, CLS similarity maps, query similarity maps, and histograms.",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"saved report: {out}")


if __name__ == "__main__":
    main()
