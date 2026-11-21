"""Per-prompt rounding thresholds tuned on dev predictions, applied to test (instead of plain rounding).

python scripts/optimize_thresholds.py --name deberta_v3_base   ->  reports/results/deberta_v3_base_thr/
Needs dev_preds_fold*.csv with a y_raw column (written by train_transformer.py).
"""
import argparse

import numpy as np
import pandas as pd

from aes.data import PROMPTS, SCORE_RANGES
from aes.metrics import qwk
from aes.tracking import RESULTS_DIR, save_predictions


def apply_cuts(raw_unit, cuts, lo):
    return lo + np.searchsorted(np.sort(cuts), raw_unit, side="right")


def fit_cuts(raw_unit, y, lo, hi, sweeps=3):
    """Coordinate ascent on the cut points between adjacent scores, starting from plain rounding."""
    n = hi - lo
    cuts = (np.arange(n) + 0.5) / n  # midpoints in [0, 1] space == plain rounding
    best = qwk(y, apply_cuts(raw_unit, cuts, lo), lo, hi)
    for _ in range(sweeps):
        for i in range(n):
            left = cuts[i - 1] if i else cuts[i] - 1.0 / n
            right = cuts[i + 1] if i < n - 1 else cuts[i] + 1.0 / n
            for c in np.linspace(left, right, 21)[1:-1]:
                trial = cuts.copy()
                trial[i] = c
                score = qwk(y, apply_cuts(raw_unit, trial, lo), lo, hi)
                if score > best:
                    best, cuts = score, trial
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    name = ap.parse_args().name
    src = RESULTS_DIR / name
    for f in sorted(src.glob("preds_fold*.csv")):
        fold = int(f.stem.removeprefix("preds_fold"))
        test, dev = pd.read_csv(f), pd.read_csv(src / f"dev_preds_fold{fold}.csv")
        out = test.copy()
        for p in PROMPTS:
            lo, hi = SCORE_RANGES[p]
            d, t = dev.prompt == p, test.prompt == p
            cuts = fit_cuts(dev.y_raw[d].to_numpy(), dev.y_true[d].to_numpy(), lo, hi)
            out.loc[t, "y_pred"] = apply_cuts(test.y_raw[t].to_numpy(), cuts, lo)
        save_predictions(out, f"{name}_thr", fold)


if __name__ == "__main__":
    main()
