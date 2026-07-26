"""Tests for the global-confusion mIoU (metric v2), probe seeding, strict loads."""

import numpy as np
import pytest
import torch

import eval_voc_dense
import utils
from eval_voc_dense import (
    compute_confusion_counts,
    compute_miou,
    file_sha256,
    load_backbone_state,
    load_dino_backbone,
    train_linear_head,
    vit_small,
)


def test_file_sha256_streams_stable_artifact_identity(tmp_path):
    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"checkpoint-fixture")

    assert file_sha256(artifact, chunk_size=3) == (
        "5d4fa22c80243cdb189dacdbf82968cd517e5153f568cb38f1592da625fdd9e0"
    )


def test_global_vs_batch_mean_hand_case():
    # Batch 1: one ignored pixel; both classes IoU 0.5 -> batch mIoU 0.5.
    pred1 = torch.tensor([0, 0, 1, 1])
    tgt1 = torch.tensor([0, 1, 1, 255])
    # Batch 2: class 0 absent entirely; class 1 perfect -> batch mIoU 1.0.
    pred2 = torch.tensor([1, 1])
    tgt2 = torch.tensor([1, 1])

    batch_mean = np.mean([
        compute_miou(pred1, tgt1, num_classes=2),
        compute_miou(pred2, tgt2, num_classes=2),
    ])
    assert abs(batch_mean - 0.75) < 1e-12

    total_inter = torch.zeros(2, dtype=torch.long)
    total_union = torch.zeros(2, dtype=torch.long)
    for pred, tgt in ((pred1, tgt1), (pred2, tgt2)):
        inter, union = compute_confusion_counts(pred, tgt, num_classes=2)
        total_inter += inter
        total_union += union
    # class 0: 1/2, class 1: 3/4 -> global mIoU 0.625 != 0.75 batch mean.
    assert total_inter.tolist() == [1, 3]
    assert total_union.tolist() == [2, 4]
    per_class = [
        total_inter[c].item() / total_union[c].item() for c in range(2) if total_union[c] > 0
    ]
    assert abs(np.mean(per_class) - 0.625) < 1e-12


def test_ignore_index_excluded_everywhere():
    pred = torch.tensor([1, 1, 0])
    tgt = torch.tensor([255, 255, 255])
    inter, union = compute_confusion_counts(pred, tgt, num_classes=2)
    assert inter.sum().item() == 0
    assert union.sum().item() == 0


def test_absent_class_union_zero():
    pred = torch.tensor([0, 0])
    tgt = torch.tensor([0, 0])
    inter, union = compute_confusion_counts(pred, tgt, num_classes=3)
    assert union[1].item() == 0 and union[2].item() == 0


def _tiny_probe_inputs():
    generator = torch.Generator().manual_seed(0)
    features_train = torch.randn(6, 4, 8, generator=generator)
    targets_train = torch.randint(0, 3, (6, 2, 2), generator=generator)
    targets_train[0, 0, 0] = 255
    features_val = torch.randn(4, 4, 8, generator=generator)
    targets_val = torch.randint(0, 3, (4, 2, 2), generator=generator)
    return features_train, targets_train, features_val, targets_val


def _run_probe(seed, *, return_stats=False):
    features_train, targets_train, features_val, targets_val = _tiny_probe_inputs()
    return train_linear_head(
        features_train, targets_train, features_val, targets_val,
        embed_dim=8, num_classes=3, patch_size=1, img_size=2,
        device=torch.device('cpu'), epochs=2, lr=0.01, batch_size=3,
        optimizer_name='adam', loss_resolution='patch', seed=seed,
        return_stats=return_stats,
    )


def test_train_linear_head_preserves_default_float_contract():
    miou = _run_probe(seed=42)
    stats = _run_probe(seed=42, return_stats=True)

    assert isinstance(miou, float)
    assert miou == stats['miou_batch_mean']


def test_train_linear_head_returns_v2_stats_when_requested_and_is_seeded():
    stats = _run_probe(seed=42, return_stats=True)
    for key in ('miou_global', 'miou_batch_mean', 'per_class_iou',
                'metric_version', 'legacy_metric_version'):
        assert key in stats
    assert stats['metric_version'] == 'global_confusion_v2'
    assert stats['legacy_metric_version'] == 'batch_mean_v1'
    assert 0.0 <= stats['miou_global'] <= 1.0
    assert 0.0 <= stats['miou_batch_mean'] <= 1.0
    assert len(stats['per_class_iou']) == 3

    repeat = _run_probe(seed=42, return_stats=True)
    assert repeat['miou_global'] == stats['miou_global']
    assert repeat['miou_batch_mean'] == stats['miou_batch_mean']


def _wrapped_backbone_state():
    reference = vit_small(patch_size=16)
    return {f'module.backbone.{k}': v for k, v in reference.state_dict().items()}


def test_strict_backbone_load_accepts_exact_match():
    model = vit_small(patch_size=16)
    msg = load_backbone_state(model, _wrapped_backbone_state(), allow_partial=False)
    assert not msg.missing_keys and not msg.unexpected_keys


def test_strict_backbone_load_rejects_missing_key():
    model = vit_small(patch_size=16)
    state = _wrapped_backbone_state()
    state.pop('module.backbone.cls_token')
    try:
        load_backbone_state(model, state, allow_partial=False)
    except RuntimeError as error:
        assert 'cls_token' in str(error)
    else:
        raise AssertionError('expected RuntimeError for missing backbone key')


def test_strict_backbone_load_allows_partial_when_requested():
    model = vit_small(patch_size=16)
    state = _wrapped_backbone_state()
    state.pop('module.backbone.cls_token')
    msg = load_backbone_state(model, state, allow_partial=True)
    assert 'cls_token' in msg.missing_keys


def test_load_dino_backbone_requires_explicit_checkpoint_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_voc_dense,
        'vit_small',
        lambda patch_size: torch.nn.Linear(2, 2),
    )
    path = tmp_path / 'checkpoint.pth'
    torch.save({'student': torch.nn.Linear(2, 2).state_dict()}, path)

    with pytest.raises(KeyError, match="teacher"):
        load_dino_backbone(path, checkpoint_key='teacher')


def test_restart_from_checkpoint_rejects_partial_module(tmp_path, monkeypatch):
    monkeypatch.delenv('DINO_ALLOW_PARTIAL_RESTORE', raising=False)
    source = torch.nn.Linear(4, 2)
    state = source.state_dict()
    del state['bias']
    path = tmp_path / 'checkpoint.pth'
    torch.save({'student': state, 'epoch': 3}, str(path))

    target = torch.nn.Linear(4, 2)
    try:
        utils.restart_from_checkpoint(str(path), run_variables=None, student=target)
    except RuntimeError as error:
        assert 'bias' in str(error)
    else:
        raise AssertionError('expected RuntimeError for partial module restore')

    monkeypatch.setenv('DINO_ALLOW_PARTIAL_RESTORE', '1')
    with pytest.warns(RuntimeWarning, match="Partially restored"):
        utils.restart_from_checkpoint(str(path), run_variables=None, student=target)


def test_restart_from_checkpoint_full_module_and_optimizer(tmp_path):
    source = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    source(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    path = tmp_path / 'checkpoint.pth'
    torch.save(
        {'student': source.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 7},
        str(path),
    )

    target = torch.nn.Linear(4, 2)
    fresh_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    restored = {'epoch': 0}
    utils.restart_from_checkpoint(
        str(path), run_variables=restored, student=target, optimizer=fresh_optimizer
    )
    assert restored['epoch'] == 7
    assert torch.equal(target.weight, source.weight)


def test_restart_from_checkpoint_propagates_optimizer_load_error(tmp_path):
    class BrokenOptimizer:
        def load_state_dict(self, state):
            raise ValueError("invalid optimizer state")

    path = tmp_path / 'checkpoint.pth'
    torch.save({'optimizer': {'state': {}}}, path)

    with pytest.raises(ValueError, match="invalid optimizer state"):
        utils.restart_from_checkpoint(
            str(path),
            run_variables=None,
            optimizer=BrokenOptimizer(),
        )
