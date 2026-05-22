import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_raw_l2_dense_eval.ipynb"


def test_kaggle_eval_notebook_exists_and_python_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        ast.parse(source, filename=f"{NOTEBOOK}:cell-{index}")


def test_kaggle_eval_notebook_uses_kaggle_paths_and_raw_l2_branch():
    source = NOTEBOOK.read_text()

    assert "/kaggle/input" in source
    assert "/kaggle/working" in source
    assert "codex/raw-l2-diagnostics" in source
    assert "analyze_patch_statistics.py" in source
    assert "fig_raw_vs_l2_dse" in source
    assert "drive.mount" not in source
