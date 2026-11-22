"""LSTM / CNN-LSTM reproduction of Taghipour & Ng (2016) on the nea folds. Fits on a 4 GB GPU.

python scripts/train_lstm.py --config configs/lstm_e50.yaml
python scripts/train_lstm.py --config configs/cnn_lstm_e50.yaml --folds 3 4   # resume only some folds
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from aes.data import PROMPTS, from_unit, load_fold, to_unit
from aes.models.lstm import predict, to_ids, train_prompt
from aes.stress import stress_variants
from aes.tracking import RESULTS_DIR, save_predictions
from aes.utils import load_config, set_seed


def frame(df, fold, raw):
    """Prediction table in the shared format (see aes.tracking), with the unrounded output kept as y_raw."""
    return pd.DataFrame({"essay_id": df.essay_id, "prompt": df.prompt, "fold": fold, "y_true": df.score,
                         "y_pred": from_unit(raw, df.prompt), "y_raw": raw})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, help="override the config, e.g. for a quick test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in args.folds:
        data = load_fold(fold)
        test = data["test"]
        variants = stress_variants(test.essay.tolist())
        # Predictions are filled prompt by prompt into these arrays, then saved once per fold.
        outs = {s: np.zeros(len(test)) for s in ["test", *variants]}
        log, t0, infer_s = [], time.time(), 0.0
        for p in PROMPTS:  # the paper trains a separate model for each prompt
            set_seed(cfg["seed"])
            tr, dv = (d[d.prompt == p].reset_index(drop=True) for d in (data["train"], data["dev"]))
            te = (test.prompt == p).to_numpy()
            model, vocab, dev_qwk, epoch = train_prompt(cfg, p, tr, dv, to_unit(tr.score, tr.prompt), device)
            t1 = time.time()
            outs["test"][te] = predict(model, to_ids(test.essay[te], vocab), device)
            infer_s += time.time() - t1  # time only the normal test predictions, for the feasibility table
            for v, texts in variants.items():
                outs[v][te] = predict(model, to_ids(np.array(texts, dtype=object)[te], vocab), device)
            log.append({"prompt": p, "best_dev_qwk": dev_qwk, "best_epoch": epoch})
            print(f"fold {fold} prompt {p}: dev QWK {dev_qwk:.3f} at epoch {epoch}")
        for split, raw in outs.items():
            save_predictions(frame(test, fold, raw), cfg["name"], fold, split if split == "test" else f"stress_{split}")
        # Run metadata (per-prompt best epochs, timing, GPU, config) for the report.
        meta = {"fold": fold, "prompts": log, "train_minutes": (time.time() - t0) / 60,
                "test_sec_per_essay": infer_s / len(test),
                "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "config": cfg}
        (RESULTS_DIR / cfg["name"] / f"meta_fold{fold}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
