from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_patch_statistics import query_points_for_grid


def test_query_points_for_grid_returns_fixed_named_points():
    points = query_points_for_grid(14, 14)

    assert [point["name"] for point in points] == [
        "center",
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
    ]
    assert [point["index"] for point in points] == [105, 45, 52, 143, 150]
