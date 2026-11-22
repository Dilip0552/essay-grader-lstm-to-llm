"""Classic baseline: hand-made features (length, vocabulary, readability) + Ridge / LightGBM, one model per prompt.

It shows how much simple surface statistics already predict, which is the reference point for the length-bias study.

Why one model per prompt? The link between features and score differs by prompt: 300 words is long for prompt 4
(~90-word answers) but short for prompt 8 (~600 words). Why standardise before Ridge? Ridge penalises large
weights equally, so features must be on comparable scales (word counts are in hundreds, ratios are below 1).
"""
import re

import numpy as np
import pandas as pd
import textstat
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_WORD = re.compile(r"[A-Za-z']+")   # a word = letters and apostrophes
_SENT = re.compile(r"[.!?]+")       # sentence boundaries


def extract_features(texts) -> pd.DataFrame:
    """One row of 12 numeric features per essay."""
    rows = []
    for t in texts:
        words = _WORD.findall(t)
        # max(..., 1) avoids division by zero on empty essays.
        n_words = max(len(words), 1)
        n_sent = max(len([s for s in _SENT.split(t) if s.strip()]), 1)
        lower = [w.lower() for w in words]
        rows.append({
            # length
            "n_words": len(words),
            "n_chars": len(t),
            "n_sentences": n_sent,
            # word and sentence complexity
            "avg_word_len": sum(map(len, words)) / n_words,
            "avg_sent_len": len(words) / n_sent,
            # vocabulary richness
            "n_unique": len(set(lower)),
            "type_token_ratio": len(set(lower)) / n_words,
            "long_word_ratio": sum(len(w) > 6 for w in words) / n_words,
            # punctuation and dataset artefacts
            "commas": t.count(","),
            "anonymized_tokens": t.count("@"),  # ASAP replaces names, places etc. with @PERSON1, @CAPS2...
            # standard readability formulas
            "flesch_reading_ease": textstat.flesch_reading_ease(t),
            "fk_grade": textstat.flesch_kincaid_grade(t),
        })
    return pd.DataFrame(rows)


def make_model(kind: str, seed: int):
    """An untrained regressor: 'ridge' (features standardised first) or 'lgbm' (gradient-boosted trees)."""
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "lgbm":
        # Small trees and subsampling, because each prompt has only ~1,000 training essays.
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    raise ValueError(f"unknown model {kind}")


def fit_predict(kind, feats_train, y_train, feats_test, seed) -> np.ndarray:
    """Train one model on the training features and return its predictions for the test features."""
    model = make_model(kind, seed).fit(feats_train, y_train)
    return model.predict(feats_test)
