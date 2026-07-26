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


def test_colab_notebook_exposes_independent_run_toggles():
    source = NOTEBOOK.read_text()

    assert "RUN_PATCH_DIAGNOSTICS = True" in source
    assert "RUN_VOC_EVAL = False" in source
    assert "RUN_PLOT_REPORT = True" in source
    assert "CREATE_OUTPUT_ARCHIVE = True" in source


def test_colab_notebook_can_run_all_eval_stages_without_runtime_patching():
    source = NOTEBOOK.read_text()
    notebook = json.loads(source)
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "eval_voc_dense.py" in source
    assert "VOC_OUTPUT_DIR = RUN_OUTPUT_ROOT / 'voc_all_checkpoints'" in source
    assert "'--ckpt_dir', str(WORK_CKPT_DIR)" in source
    assert "'--output_dir', str(VOC_OUTPUT_DIR)" in source
    assert "'--optimizer', VOC_OPTIMIZER" in source
    assert "'--probe_seed', str(VOC_PROBE_SEED)" in source
    assert "'--checkpoint_key', VOC_CHECKPOINT_KEY" in source
    assert "VOC_CHECKPOINT_KEY == CHECKPOINT_KEY" in source
    assert code_source.index("VOC_CHECKPOINT_KEY = 'teacher'") < code_source.index(
        "assert VOC_CHECKPOINT_KEY == CHECKPOINT_KEY"
    )
    assert code_source.index("\nCHECKPOINT_KEY = 'teacher'") < code_source.index(
        "assert VOC_CHECKPOINT_KEY == CHECKPOINT_KEY"
    )
    assert "voc_miou_results_global_confusion_v2.json" in source
    assert "'--voc_protocol', VOC_PROTOCOL" in source
    assert "'--voc_metric_version', VOC_METRIC_VERSION" in source
    assert "'--voc_probe_seed', str(VOC_PROBE_SEED)" in source
    assert "'--voc_checkpoint_key', VOC_CHECKPOINT_KEY" in source
    assert "txt.replace(old, new)" not in source
    assert "optimizer patch" not in source.lower()
