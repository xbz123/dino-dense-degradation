from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dense_results_io import read_csv_rows, read_voc_results, to_number


def test_to_number_parses_numeric_strings_and_preserves_text():
    assert to_number("12") == 12
    assert to_number("12.5") == 12.5
    assert to_number("") != to_number("")
    assert to_number("epoch") == "epoch"


def test_read_csv_rows_sorts_by_epoch(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("epoch,value\n30,1.5\n20,1.0\n")

    rows = read_csv_rows(path)

    assert [row["epoch"] for row in rows] == [20, 30]
    assert rows[0]["value"] == 1.0


def test_read_voc_results_can_return_mapping_or_rows(tmp_path):
    path = tmp_path / "voc.json"
    path.write_text(json.dumps([{"epoch": 30, "miou": 31.0}, {"epoch": 20, "miou": 30.0}]))

    assert read_voc_results(path) == {20: 30.0, 30: 31.0}
    assert [row["epoch"] for row in read_voc_results(path, as_rows=True)] == [20, 30]
