"""The single place where experiments store their results.

Every experiment writes predictions to reports/results/<name>/. Tables, figures and the report are built only from
these files, never from numbers typed by hand. Prediction files hold essay ids and scores only, never essay text
(the ASAP competition rules forbid sharing the essays).
"""
import pandas as pd

from aes.data import ROOT
from aes.metrics import per_prompt_qwk

RESULTS_DIR = ROOT / "reports" / "results"
# Columns every prediction file must have; models may add more (e.g. y_raw, the unrounded output).
PRED_COLS = ["essay_id", "prompt", "fold", "y_true", "y_pred"]


def save_predictions(preds: pd.DataFrame, name: str, fold: int, split: str = "test") -> pd.Series:
    """Save one fold's predictions (columns PRED_COLS, y_pred as integer score; extra columns kept); return per-prompt QWK.
    split="test" goes into the results table; other splits (dev, stress tests) are saved as <split>_preds_fold{k}.csv."""
    missing = set(PRED_COLS) - set(preds.columns)
    if missing:
        raise ValueError(f"predictions missing columns {missing}")
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    # Only files named preds_fold*.csv are picked up by scripts/make_tables.py.
    preds.to_csv(out / (f"preds_fold{fold}.csv" if split == "test" else f"{split}_preds_fold{fold}.csv"), index=False)
    scores = per_prompt_qwk(preds)
    print(f"[{name} fold {fold} {split}] avg QWK {scores.mean():.3f} | " + " ".join(f"P{p}={q:.3f}" for p, q in scores.items()))
    return scores
