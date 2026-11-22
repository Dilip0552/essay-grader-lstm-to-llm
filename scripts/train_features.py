"""Feature baselines on the nea folds (CPU, ~3 min). Runs every experiment listed in the config.

python scripts/train_features.py --config configs/features.yaml
"""
import argparse

import numpy as np
import pandas as pd

from aes.data import PROMPTS, from_unit, load_fold, to_unit
from aes.models.features import extract_features, fit_predict
from aes.stress import stress_variants
from aes.tracking import save_predictions
from aes.utils import load_config, set_seed


def frame(df, fold, raw):
    """Prediction table in the shared format (see aes.tracking), with the unrounded output kept as y_raw."""
    return pd.DataFrame({"essay_id": df.essay_id, "prompt": df.prompt, "fold": fold, "y_true": df.score,
                         "y_pred": from_unit(raw, df.prompt), "y_raw": raw})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)

    for fold in range(5):
        data = load_fold(fold)
        test = data["test"]
        # Features are computed once per fold and shared by all experiments.
        feats = {k: extract_features(v.essay) for k, v in data.items()}
        stress = {k: extract_features(t) for k, t in stress_variants(test.essay.tolist()).items()}
        y = to_unit(data["train"].score, data["train"].prompt)
        for name, exp in cfg["experiments"].items():
            set_seed(cfg["seed"])
            cols = list(feats["train"].columns) if exp["features"] == "all" else exp["features"]
            outs = {s: np.zeros(len(test)) for s in ["test", *stress]}
            for p in PROMPTS:  # one model per prompt
                tr, te = (data["train"].prompt == p).to_numpy(), (test.prompt == p).to_numpy()
                # Same training data for the normal test set and each stress-test version of it.
                for split, f in [("test", feats["test"]), *stress.items()]:
                    outs[split][te] = fit_predict(exp["model"], feats["train"].loc[tr, cols], y[tr], f.loc[te, cols], cfg["seed"])
            for split, raw in outs.items():
                save_predictions(frame(test, fold, raw), name, fold, split if split == "test" else f"stress_{split}")


if __name__ == "__main__":
    main()
