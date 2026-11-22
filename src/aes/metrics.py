"""Quadratic Weighted Kappa (QWK), the standard agreement metric for ASAP.

QWK is 1 for perfect agreement and 0 for agreement at chance level. Disagreements are penalised by the squared
distance between the two scores, so being 3 points off costs 9 times more than being 1 point off.

Why QWK and not accuracy? Scores are ordered, and accuracy treats "off by 1" the same as "off by 5". QWK also
corrects for chance agreement, and it was the official metric of the ASAP competition and of the papers we
compare with.
"""
import numpy as np
import pandas as pd

from aes.data import SCORE_RANGES


def qwk(y_true, y_pred, min_rating=None, max_rating=None) -> float:
    """Quadratic weighted kappa on integer ratings in [min_rating, max_rating]."""
    a = np.asarray(y_true, dtype=int)
    b = np.asarray(y_pred, dtype=int)
    # Pass the prompt's official range so unused score points still count; otherwise use the range in the data.
    lo = min(a.min(), b.min()) if min_rating is None else min_rating
    hi = max(a.max(), b.max()) if max_rating is None else max_rating
    n = hi - lo + 1
    if n == 1:
        return 1.0
    # Observed agreement: how often true score i was predicted as j.
    observed = np.zeros((n, n))
    np.add.at(observed, (a - lo, b - lo), 1)
    # Agreement expected by chance if both raters kept their score distributions but were independent.
    expected = np.outer(observed.sum(1), observed.sum(0)) / len(a)
    # Quadratic penalty: 0 on the diagonal, 1 for the largest possible disagreement.
    i, j = np.indices((n, n))
    w = (i - j) ** 2 / (n - 1) ** 2
    denom = (w * expected).sum()
    return 1.0 if denom == 0 else 1.0 - (w * observed).sum() / denom  # denom 0 only when both sides share one category


def per_prompt_qwk(preds: pd.DataFrame) -> pd.Series:
    """QWK per prompt, averaged over folds (nea protocol). Needs columns prompt, fold, y_true, y_pred."""
    rows = [(p, qwk(g.y_true, g.y_pred, *SCORE_RANGES[p])) for (_, p), g in preds.groupby(["fold", "prompt"])]
    return pd.DataFrame(rows, columns=["prompt", "qwk"]).groupby("prompt").qwk.mean()
