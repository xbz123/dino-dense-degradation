from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dense_patch_diagnostics import (
    attention_stats,
    class_separability,
    cls_patch_cosine_stats,
    fit_pca_projector,
    fixed_query_similarity_stats,
    normalize01,
    project_pca_rgb,
    spectrum_metrics,
)


def test_spectrum_metrics_reports_low_rank_structure():
    x = torch.zeros(24, 4)
    x[:, 0] = torch.linspace(-1, 1, 24)

    metrics = spectrum_metrics(x)

    assert metrics["num_tokens"] == 24
    assert metrics["feature_dim"] == 4
    assert metrics["effective_rank"] < 1.2
    assert metrics["top1_eigen_ratio"] > 0.99


def test_cls_patch_cosine_stats_uses_all_patch_tokens():
    cls = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    patches = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, -1.0]],
        ]
    )

    stats = cls_patch_cosine_stats(cls, patches)

    assert stats["cls_patch_cos_mean"] == 0.25
    assert stats["cls_patch_cos_p90"] <= 1.0
    assert stats["cls_patch_cos_min"] >= -1.0


def test_attention_stats_reports_entropy_and_top_mass():
    attn = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])

    stats = attention_stats(attn, topk=2)

    assert stats["cls_attention_mean"] == pytest.approx(0.25)
    assert stats["cls_attention_max_mean"] == pytest.approx(0.475)
    assert stats["cls_attention_top2_mass_mean"] == pytest.approx(0.65)
    assert stats["cls_attention_entropy_mean"] > 0.0


def test_class_separability_detects_two_obvious_clusters():
    tokens = torch.tensor(
        [
            [[1.0, 0.0], [1.1, 0.0], [-1.0, 0.0], [-1.1, 0.0]],
            [[0.9, 0.0], [1.2, 0.0], [-0.9, 0.0], [-1.2, 0.0]],
        ]
    )

    metrics = class_separability(tokens, k=2, seed=0)

    assert metrics["class_sep_k2"] > 1.0
    assert metrics["minter_k2"] > metrics["mintra_k2"]


def test_fixed_query_similarity_stats_has_expected_entropy_and_correlation():
    patches = torch.eye(4).reshape(1, 4, 4)
    stats = fixed_query_similarity_stats(
        patches,
        query_indices=[0, 2],
        previous_maps=None,
        prefix="query",
    )
    stats_with_prev = fixed_query_similarity_stats(
        patches,
        query_indices=[0, 2],
        previous_maps=stats["query_similarity_maps"],
        prefix="query",
    )

    assert stats["query_sim_entropy_mean"] > 0
    assert stats["query_sim_top10pct_mass_mean"] > 0
    assert stats_with_prev["query_sim_early_corr_mean"] == 1.0


def test_normalize01_handles_constant_arrays():
    out = normalize01(np.full((2, 2), 7.0))

    assert out.shape == (2, 2)
    assert np.all(out == 0.0)


def test_fixed_pca_projector_uses_shared_basis_and_range():
    early = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    late = early + torch.tensor([2.0, 0.0, 0.0])

    projector = fit_pca_projector([early, late])
    early_rgb = project_pca_rgb(early, projector, 2, 2)
    late_rgb = project_pca_rgb(late, projector, 2, 2)

    assert early_rgb.shape == (2, 2, 3)
    assert late_rgb.shape == (2, 2, 3)
    assert projector.basis.shape == (3, 3)
    assert not np.allclose(early_rgb, late_rgb)
