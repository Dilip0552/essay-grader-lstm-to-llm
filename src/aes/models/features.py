"""Hand-made features (length, vocabulary, readability) + Ridge / LightGBM, one model per prompt."""
import re

import numpy as np
import pandas as pd
import textstat
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[.!?]+")


def extract_features(texts) -> pd.DataFrame:
    rows = []
    for t in texts:
        words = _WORD.findall(t)
        n_words = max(len(words), 1)
        n_sent = max(len([s for s in _SENT.split(t) if s.strip()]), 1)
        lower = [w.lower() for w in words]
        rows.append({
            "n_words": len(words),
            "n_chars": len(t),
            "n_sentences": n_sent,
            "avg_word_len": sum(map(len, words)) / n_words,
            "avg_sent_len": len(words) / n_sent,
            "n_unique": len(set(lower)),
            "type_token_ratio": len(set(lower)) / n_words,
            "long_word_ratio": sum(len(w) > 6 for w in words) / n_words,
            "commas": t.count(","),
            "anonymized_tokens": t.count("@"),  # ASAP replaces names, places etc. with @PERSON1, @CAPS2...
            "flesch_reading_ease": textstat.flesch_reading_ease(t),
            "fk_grade": textstat.flesch_kincaid_grade(t),
        })
    return pd.DataFrame(rows)


def make_model(kind: str, seed: int):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "lgbm":
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    raise ValueError(f"unknown model {kind}")


def fit_predict(kind, feats_train, y_train, feats_test, seed) -> np.ndarray:
    model = make_model(kind, seed).fit(feats_train, y_train)
    return model.predict(feats_test)
