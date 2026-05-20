"""Tests for the pointwise/pairwise re-ranking losses."""

from __future__ import annotations

import numpy as np

from src.reranking.cross_encoder import pairwise_margin_loss, pointwise_bce_loss


def test_bce_loss_zero_at_perfect_prediction() -> None:
    # Very high positive logits with label 1 ⇒ ~0 loss
    scores = np.array([10.0, 10.0, 10.0])
    labels = np.array([1.0, 1.0, 1.0])
    assert pointwise_bce_loss(scores, labels) < 1e-3


def test_bce_loss_high_at_wrong_prediction() -> None:
    scores = np.array([-10.0, -10.0, -10.0])
    labels = np.array([1.0, 1.0, 1.0])
    assert pointwise_bce_loss(scores, labels) > 5.0


def test_pairwise_margin_zero_when_well_separated() -> None:
    pos = np.array([3.0, 3.0])
    neg = np.array([0.0, 0.0])
    assert pairwise_margin_loss(pos, neg, margin=1.0) == 0.0


def test_pairwise_margin_positive_when_inverted() -> None:
    pos = np.array([0.0, 0.0])
    neg = np.array([2.0, 2.0])
    assert pairwise_margin_loss(pos, neg, margin=1.0) > 0.0
