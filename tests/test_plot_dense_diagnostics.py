from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dense_results_io import read_csv_rows
from plot_dense_diagnostics import plot_raw_l2_figures, write_combined_summary


def test_plot_raw_l2_figures_writes_required_outputs(tmp_path):
    rows = [
        {
            "epoch": 1,
            "raw_dse": 10.0,
            "l2_dse": 8.0,
            "raw_class_sep_avg": -4.0,
            "l2_class_sep_avg": -1.0,
            "raw_effective_rank": 10.0,
            "l2_effective_rank": 5.0,
            "raw_top1_eigen_ratio": 0.4,
            "l2_top1_eigen_ratio": 0.2,
        },
        {
            "epoch": 2,
            "raw_dse": 6.0,
            "l2_dse": 9.0,
            "raw_class_sep_avg": -8.0,
            "l2_class_sep_avg": -1.5,
            "raw_effective_rank": 20.0,
            "l2_effective_rank": 4.0,
            "raw_top1_eigen_ratio": 0.2,
            "l2_top1_eigen_ratio": 0.3,
        },
    ]

    written = plot_raw_l2_figures(rows, tmp_path)

    assert [path.name for path in written] == [
        "fig_raw_vs_l2_dse.png",
        "fig_raw_vs_l2_class_sep.png",
        "fig_raw_vs_l2_spectrum.png",
    ]
    assert all(path.is_file() for path in written)


def test_combined_summary_preserves_raw_l2_columns(tmp_path):
    rows = [
        {
            "epoch": 1,
            "raw_dse": 10.0,
            "l2_dse": 8.0,
        }
    ]
    path = tmp_path / "combined.csv"

    write_combined_summary(path, rows, {1: 33.0})

    reloaded = read_csv_rows(path)
    assert reloaded[0]["voc_miou"] == 33.0
    assert reloaded[0]["raw_dse"] == 10.0
    assert reloaded[0]["l2_dse"] == 8.0
