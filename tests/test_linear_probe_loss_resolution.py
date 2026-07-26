import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval_voc_dense


def test_set_probe_seed_replays_all_probe_rngs():
    eval_voc_dense.set_probe_seed(2027)
    first = (
        random.random(),
        np.random.rand(3),
        torch.rand(3),
    )

    eval_voc_dense.set_probe_seed(2027)
    second = (
        random.random(),
        np.random.rand(3),
        torch.rand(3),
    )

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_train_linear_head_patch_loss_uses_patch_grid_targets(monkeypatch):
    features_train = torch.randn(1, 4, 3)
    features_val = torch.randn(1, 4, 3)
    targets_train = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]])
    targets_val = targets_train.clone()
    loss_shapes = []

    def fake_cross_entropy(logits, target, ignore_index):
        loss_shapes.append((tuple(logits.shape), tuple(target.shape), ignore_index))
        return logits.sum() * 0.0

    monkeypatch.setattr(eval_voc_dense.F, "cross_entropy", fake_cross_entropy)

    miou = eval_voc_dense.train_linear_head(
        features_train,
        targets_train,
        features_val,
        targets_val,
        embed_dim=3,
        num_classes=2,
        patch_size=2,
        img_size=4,
        device=torch.device("cpu"),
        epochs=1,
        lr=0.01,
        batch_size=1,
        optimizer_name="sgd",
        loss_resolution="patch",
    )

    assert loss_shapes == [((1, 2, 2, 2), (1, 2, 2), 255)]
    assert math.isfinite(miou)
