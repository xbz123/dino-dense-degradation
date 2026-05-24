from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_dino_accepts_explicit_resume_checkpoint_argument():
    source = (ROOT / "main_dino.py").read_text()

    assert "--resume_from" in source
    assert "args.resume_from or os.path.join(args.output_dir, \"checkpoint.pth\")" in source


def test_colab_continue_uses_checkpoint_argument_without_copying_over_output_checkpoint():
    source = (ROOT / "colab_continue_200.sh").read_text()

    assert 'RESUME_CKPT="${1:-/content/drive/MyDrive/dinocheckpoint/checkpoint170.pth}"' in source
    assert "--resume_from \"${RESUME_CKPT}\"" in source
    assert "--saveckp_freq 10" in source
    assert "--keep_last_ckpts 0" in source
    assert 'cp "${LATEST_CKPT}" "${RESUME_CKPT}"' not in source
    assert 'cp "${RESUME_CKPT}" "${OUTPUT_DIR}/checkpoint.pth"' not in source


def test_colab_setup_preserves_every_ten_epoch_checkpoint():
    source = (ROOT / "colab_setup.sh").read_text()

    assert "--saveckp_freq 10" in source
    assert "--keep_last_ckpts 0" in source


def test_kaggle_guide_uses_current_resume_from_workflow():
    source = (ROOT / "KAGGLE_GUIDE.md").read_text()

    assert "--resume_from" in source
    assert "--saveckp_freq 10" in source
    assert "--keep_last_ckpts 0" in source
    assert "code.replace(" not in source
    assert "shutil.copy(old_ckpt_path" not in source
    assert "dinocehckpoint" not in source


def test_eval_voc_dense_path_hint_matches_current_drive_checkpoint_folder():
    source = (ROOT / "eval_voc_dense.py").read_text()

    assert "/content/drive/MyDrive/dinocheckpoint" in source
    assert "dinocehckpoint" not in source


def test_eval_voc_dense_exposes_optimizer_choice_for_notebooks():
    source = (ROOT / "eval_voc_dense.py").read_text()

    assert "--optimizer" in source
    assert "choices=['adam', 'sgd']" in source
    assert "optimizer_name=args.optimizer" in source
