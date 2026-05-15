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
    assert 'cp "${LATEST_CKPT}" "${RESUME_CKPT}"' not in source
    assert 'cp "${RESUME_CKPT}" "${OUTPUT_DIR}/checkpoint.pth"' not in source
