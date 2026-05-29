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


def test_kaggle_eval_notebook_defaults_to_full_raw_l2_sweep():
    source = NOTEBOOK.read_text()

    assert "CHECKPOINT_EPOCH_FILTER = None" in source
    assert "NUM_DSE_IMAGES = 2048" in source
    assert "NUM_VIS_IMAGES = 6" in source
    assert "PATCH_DSE_GROUP_STRIDE = 1" in source
    assert "OUTPUT_RUN_SUFFIX = 'raw_l2_full'" in source


def test_kaggle_eval_notebook_keeps_checkpoint_copies_out_of_saved_output():
    source = NOTEBOOK.read_text()

    assert "WORK_CKPT_DIR = Path('/tmp/dino_eval_checkpoints')" in source
    assert "WORK_CKPT_DIR = Path('/kaggle/working/dino_eval_checkpoints')" not in source


def test_kaggle_eval_notebook_can_reuse_any_attached_voc_json():
    source = NOTEBOOK.read_text()

    assert "VOC_JSON_CANDIDATES.extend" in source
    assert "glob('to_epoch_*/voc_all_checkpoints/voc_miou_results.json')" in source


def test_kaggle_eval_notebook_exposes_independent_run_toggles():
    source = NOTEBOOK.read_text()

    assert "RUN_PATCH_DIAGNOSTICS = True" in source
    assert "RUN_VOC_EVAL = False" in source
    assert "RUN_PLOT_REPORT = True" in source
    assert "CREATE_OUTPUT_ARCHIVE = True" in source


def test_kaggle_eval_notebook_can_run_voc_when_enabled():
    source = NOTEBOOK.read_text()

    assert "NotImplementedError" not in source
    assert "eval_voc_dense.py" in source
    assert "VOC_OUTPUT_DIR = RUN_OUTPUT_ROOT / 'voc_all_checkpoints'" in source
    assert "'--ckpt_dir', str(WORK_CKPT_DIR)" in source
    assert "'--output_dir', str(VOC_OUTPUT_DIR)" in source
    assert "'--optimizer', VOC_OPTIMIZER" in source
