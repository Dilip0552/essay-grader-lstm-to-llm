import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from aes.metrics import per_prompt_qwk, qwk


def test_perfect_agreement():
    assert qwk([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 7, 300), rng.integers(0, 7, 300)
    assert qwk(a, b) == pytest.approx(cohen_kappa_score(a, b, weights="quadratic"))


def test_unused_categories_in_range_change_nothing_when_absent_from_both():
    a, b = [1, 2, 3, 2], [1, 3, 3, 2]
    assert qwk(a, b, 1, 3) == pytest.approx(qwk(a, b))


def test_known_value():
    # Hand-computed: O=[[1,1],[0,2]], E=[[0.5,1.5],[0.5,1.5]] -> 1 - 1/2 = 0.5
    assert qwk([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)


def test_all_same_category():
    assert qwk([2, 2, 2], [2, 2, 2], 0, 3) == 1.0


def test_per_prompt_averages_over_folds():
    df = pd.DataFrame({"prompt": [3] * 8, "fold": [0] * 4 + [1] * 4,
                       "y_true": [0, 1, 2, 3, 0, 0, 1, 1], "y_pred": [0, 1, 2, 3, 0, 1, 1, 1]})
    expected = (1.0 + qwk([0, 0, 1, 1], [0, 1, 1, 1], 0, 3)) / 2
    assert per_prompt_qwk(df)[3] == pytest.approx(expected)


def test_threshold_tuning_never_worse_than_rounding_on_dev():
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
