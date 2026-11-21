"""ASAP loading, the nea 5-fold splits (Taghipour & Ng 2016), and per-prompt score scaling."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FOLDS_DIR = DATA_DIR / "folds"

# domain1_score range per prompt, as in nea's asap_reader.py
SCORE_RANGES = {1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)}
PROMPTS = sorted(SCORE_RANGES)
# In the nea fold files but absent from the current Kaggle training_set_rel3.tsv (12,976 essays vs nea's 12,978)
NEA_IDS_NOT_ON_KAGGLE = {10535, 10536}

_COLS = ["essay_id", "essay_set", "essay", "rater1_domain1", "rater2_domain1", "domain1_score"]


def tsv_path() -> Path:
    return Path(os.environ.get("ASAP_TSV") or DATA_DIR / "training_set_rel3.tsv")


def load_asap(path=None) -> pd.DataFrame:
    """All essays with columns essay_id, prompt, essay, rater1, rater2, score."""
    path = Path(path or tsv_path())
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README section 'Data'.")
    df = pd.read_csv(path, sep="\t", encoding="latin-1", usecols=_COLS)
    df = df.rename(columns={"essay_set": "prompt", "domain1_score": "score",
                            "rater1_domain1": "rater1", "rater2_domain1": "rater2"})
    lo, hi = _bounds(df.prompt)
    bad = (df.score < lo) | (df.score > hi)
    if bad.any():
        raise ValueError(f"{bad.sum()} scores outside their prompt's range, e.g. essay_id {df.essay_id[bad].iloc[0]}")
    return df


def load_fold(fold: int, path=None, folds_dir=None) -> dict:
    """{'train','dev','test'} DataFrames for one of the 5 nea folds."""
    df = load_asap(path)
    folds_dir = Path(folds_dir or FOLDS_DIR)
    out = {}
    for split in ("train", "dev", "test"):
        ids = np.loadtxt(folds_dir / f"fold_{fold}" / f"{split}_ids.txt", dtype=int, ndmin=1)
        part = df[df.essay_id.isin(ids)].reset_index(drop=True)
        missing = set(ids) - set(part.essay_id) - NEA_IDS_NOT_ON_KAGGLE
        if missing:
            raise ValueError(f"fold {fold} {split}: ids not in the TSV: {sorted(missing)[:5]}")
        out[split] = part
    return out


def _bounds(prompts):
    p = np.asarray(prompts)
    lo = np.array([SCORE_RANGES[x][0] for x in p], dtype=float)
    hi = np.array([SCORE_RANGES[x][1] for x in p], dtype=float)
    return lo, hi


def to_unit(scores, prompts) -> np.ndarray:
    """Scale scores to [0, 1] using each essay's prompt range."""
    lo, hi = _bounds(prompts)
    return (np.asarray(scores, dtype=float) - lo) / (hi - lo)


def from_unit(x, prompts) -> np.ndarray:
    """Map [0, 1] predictions back to integer scores in each prompt's range."""
    lo, hi = _bounds(prompts)
    return np.clip(np.rint(lo + np.asarray(x, dtype=float) * (hi - lo)), lo, hi).astype(int)
