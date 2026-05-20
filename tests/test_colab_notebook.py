import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_dense_degradation_all_checkpoints.ipynb"


def test_colab_notebook_python_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        ast.parse(source, filename=f"{NOTEBOOK}:cell-{index}")


def test_colab_notebook_calls_repository_diagnostic_scripts():
    source = NOTEBOOK.read_text()

    assert "analyze_patch_statistics.py" in source
    assert "plot_dense_diagnostics.py" in source
    assert "make_summary_report.py" in source
    assert "patch_attention_dse_suite.py" not in source
