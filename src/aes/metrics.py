import numpy as np
import pandas as pd

from aes.data import SCORE_RANGES


def qwk(y_true, y_pred, min_rating=None, max_rating=None) -> float:
    """Quadratic weighted kappa on integer ratings in [min_rating, max_rating]."""
    a = np.asarray(y_true, dtype=int)
    b = np.asarray(y_pred, dtype=int)
    lo = min(a.min(), b.min()) if min_rating is None else min_rating
    hi = max(a.max(), b.max()) if max_rating is None else max_rating
    n = hi - lo + 1
    if n == 1:
        return 1.0
    observed = np.zeros((n, n))
    np.add.at(observed, (a - lo, b - lo), 1)
    expected = np.outer(observed.sum(1), observed.sum(0)) / len(a)
    i, j = np.indices((n, n))
    w = (i - j) ** 2 / (n - 1) ** 2
    denom = (w * expected).sum()
    return 1.0 if denom == 0 else 1.0 - (w * observed).sum() / denom  # denom 0 only when both sides share one category


def per_prompt_qwk(preds: pd.DataFrame) -> pd.Series:
    """QWK per prompt, averaged over folds (nea protocol). Needs columns prompt, fold, y_true, y_pred."""
    rows = [(p, qwk(g.y_true, g.y_pred, *SCORE_RANGES[p])) for (_, p), g in preds.groupby(["fold", "prompt"])]
    return pd.DataFrame(rows, columns=["prompt", "qwk"]).groupby("prompt").qwk.mean()
