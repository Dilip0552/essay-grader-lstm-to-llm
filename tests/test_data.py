"""Tests for data loading, fold splitting and score scaling, on a tiny fake TSV (the real data is not in the repo)."""
import numpy as np
import pandas as pd
import pytest

from aes.data import SCORE_RANGES, from_unit, load_asap, load_fold, to_unit


@pytest.fixture
def fake_asap(tmp_path):
    """A 6-essay TSV with the real column layout, plus fold id files, written to a temporary folder."""
    rows = [(i, p, f"essay {i}", 1, 1, SCORE_RANGES[p][0] + i % 2) for i, p in enumerate([1, 2, 7, 8, 3, 5], start=1)]
    df = pd.DataFrame(rows, columns=["essay_id", "essay_set", "essay", "rater1_domain1", "rater2_domain1", "domain1_score"])
    df["extra_col"] = 0  # real TSV has many more columns
    tsv = tmp_path / "asap.tsv"
    df.to_csv(tsv, sep="\t", index=False, encoding="latin-1")
    folds = tmp_path / "folds" / "fold_0"
    folds.mkdir(parents=True)
    for split, ids in {"train": [1, 2, 3, 4], "dev": [5], "test": [6]}.items():
        (folds / f"{split}_ids.txt").write_text("\n".join(map(str, ids)) + "\n")
    return tsv, tmp_path / "folds"


def test_load_asap_renames_and_keeps_all(fake_asap):
    """Columns are renamed to our short names and no essay is lost."""
    df = load_asap(fake_asap[0])
    assert len(df) == 6
    assert {"essay_id", "prompt", "essay", "rater1", "rater2", "score"} <= set(df.columns)


def test_load_fold_splits_are_disjoint_and_complete(fake_asap):
    """No essay may appear in two splits: that would leak test answers into training."""
    f = load_fold(0, *fake_asap)
    ids = [set(f[s].essay_id) for s in ("train", "dev", "test")]
    assert [len(x) for x in ids] == [4, 1, 1]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_load_fold_rejects_unknown_ids(fake_asap):
    """A fold id missing from the TSV means data and splits do not match: fail loudly."""
    tsv, folds = fake_asap
    (folds / "fold_0" / "test_ids.txt").write_text("999\n")
    with pytest.raises(ValueError):
        load_fold(0, tsv, folds)


def test_out_of_range_score_rejected(tmp_path, fake_asap):
    """A score outside the prompt's range signals a corrupt file or wrong ranges."""
    df = pd.read_csv(fake_asap[0], sep="\t")
    df.loc[0, "domain1_score"] = 99
    bad = tmp_path / "bad.tsv"
    df.to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError):
        load_asap(bad)


@pytest.mark.parametrize("prompt", sorted(SCORE_RANGES))
def test_unit_roundtrip_every_prompt(prompt):
    """Scaling to [0, 1] and back must return exactly the original score for every prompt."""
    lo, hi = SCORE_RANGES[prompt]
    scores = np.arange(lo, hi + 1)
    prompts = [prompt] * len(scores)
    u = to_unit(scores, prompts)
    assert u.min() == 0 and u.max() == 1
    assert (from_unit(u, prompts) == scores).all()


def test_from_unit_clips():
    """Model outputs slightly outside [0, 1] must still map to a valid score."""
    assert list(from_unit([-0.5, 1.7], [8, 8])) == [0, 60]
