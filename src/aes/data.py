"""Data access for the whole project.

- Loads the ASAP essays from Kaggle's training_set_rel3.tsv.
- Splits them with the 5 folds published by Taghipour & Ng (2016) in the nusnlp/nea repository, so our numbers are
  comparable with the paper.
- Converts scores to and from [0, 1]. Every prompt has a different score range, so models learn one common target
  scale and predictions are mapped back per prompt.

Why fixed published folds instead of our own random split? Results then compare directly with Taghipour & Ng and
with later papers that use the same folds; a different split would make our QWK numbers incomparable.
Why train/dev/test? Train fits the model, dev chooses the best epoch and the rounding thresholds, and test is used
only once for the reported numbers, so they are not optimistically biased.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root, resolved from this file's location so scripts work from any working directory.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"          # gitignored: the dataset is never committed
FOLDS_DIR = DATA_DIR / "folds"    # filled by scripts/download_data.py

# Official range of the final score (domain1_score) for each of the 8 prompts, as in nea's asap_reader.py.
SCORE_RANGES = {1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)}
PROMPTS = sorted(SCORE_RANGES)
# In the nea fold files but absent from the current Kaggle training_set_rel3.tsv (12,976 essays vs nea's 12,978)
NEA_IDS_NOT_ON_KAGGLE = {10535, 10536}

# Only the columns we use; the TSV has many more (trait scores, rater 3, ...).
_COLS = ["essay_id", "essay_set", "essay", "rater1_domain1", "rater2_domain1", "domain1_score"]


def tsv_path() -> Path:
    """Location of the ASAP TSV: the ASAP_TSV environment variable (set on Kaggle/Colab) or data/ locally."""
    return Path(os.environ.get("ASAP_TSV") or DATA_DIR / "training_set_rel3.tsv")


def load_asap(path=None) -> pd.DataFrame:
    """All essays with columns essay_id, prompt, essay, rater1, rater2, score."""
    path = Path(path or tsv_path())
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README section 'Data'.")
    # The file is not valid UTF-8; latin-1 reads every byte without errors.
    df = pd.read_csv(path, sep="\t", encoding="latin-1", usecols=_COLS)
    df = df.rename(columns={"essay_set": "prompt", "domain1_score": "score",
                            "rater1_domain1": "rater1", "rater2_domain1": "rater2"})
    # Fail loudly if a score is outside its prompt's official range: that would mean a wrong file or wrong ranges.
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
        # Each split file is a plain list of essay ids, one per line.
        ids = np.loadtxt(folds_dir / f"fold_{fold}" / f"{split}_ids.txt", dtype=int, ndmin=1)
        part = df[df.essay_id.isin(ids)].reset_index(drop=True)
        # Any id we cannot find (other than the 2 known ones) means the data and splits do not match.
        missing = set(ids) - set(part.essay_id) - NEA_IDS_NOT_ON_KAGGLE
        if missing:
            raise ValueError(f"fold {fold} {split}: ids not in the TSV: {sorted(missing)[:5]}")
        out[split] = part
    return out


def _bounds(prompts):
    """Per-essay lower and upper score bounds, as float arrays aligned with `prompts`."""
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
    # Round to the nearest valid score and clip, since a regression output can fall slightly outside [0, 1].
    return np.clip(np.rint(lo + np.asarray(x, dtype=float) * (hi - lo)), lo, hi).astype(int)
