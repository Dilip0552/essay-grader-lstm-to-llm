"""Tests for the QWK metric: it must match scikit-learn and behave correctly on edge cases, because every result in the project depends on it."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from aes.metrics import per_prompt_qwk, qwk


def test_perfect_agreement():
    """Identical ratings must give the maximum, 1."""
    assert qwk([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_matches_sklearn():
    """Our implementation must agree with scikit-learn's well-tested cohen_kappa_score."""
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 7, 300), rng.integers(0, 7, 300)
    assert qwk(a, b) == pytest.approx(cohen_kappa_score(a, b, weights="quadratic"))


def test_unused_categories_in_range_change_nothing_when_absent_from_both():
    """Passing the official range must not change the result when no score is outside the observed range."""
    a, b = [1, 2, 3, 2], [1, 3, 3, 2]
    assert qwk(a, b, 1, 3) == pytest.approx(qwk(a, b))


def test_known_value():
    """A small case worked out by hand, so the formula itself is checked, not just consistency."""
    # Hand-computed: O=[[1,1],[0,2]], E=[[0.5,1.5],[0.5,1.5]] -> 1 - 1/2 = 0.5
    assert qwk([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)


def test_all_same_category():
    """Edge case: everyone gets the same score; must not divide by zero."""
    assert qwk([2, 2, 2], [2, 2, 2], 0, 3) == 1.0


def test_per_prompt_averages_over_folds():
    """QWK is computed per fold and then averaged (the nea protocol), not on all folds pooled."""
    df = pd.DataFrame({"prompt": [3] * 8, "fold": [0] * 4 + [1] * 4,
                       "y_true": [0, 1, 2, 3, 0, 0, 1, 1], "y_pred": [0, 1, 2, 3, 0, 1, 1, 1]})
    expected = (1.0 + qwk([0, 0, 1, 1], [0, 1, 1, 1], 0, 3)) / 2
    assert per_prompt_qwk(df)[3] == pytest.approx(expected)


def test_threshold_tuning_never_worse_than_rounding_on_dev():
    """Tuned cut points start from plain rounding and only accept improvements, so dev QWK can never get worse."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from optimize_thresholds import apply_cuts, fit_cuts

    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    raw = np.clip(y / 3 * 0.7 + 0.1 + rng.normal(0, 0.08, 400), 0, 1)  # compressed predictions: rounding is biased
    plain = (np.arange(3) + 0.5) / 3
    cuts = fit_cuts(raw, y, 0, 3)
    assert qwk(y, apply_cuts(raw, cuts, 0), 0, 3) >= qwk(y, apply_cuts(raw, plain, 0), 0, 3)
    assert list(apply_cuts(np.array([0.0, 0.2, 0.5, 0.9]), plain, 0)) == [0, 1, 2, 3]  # midpoints == rounding
