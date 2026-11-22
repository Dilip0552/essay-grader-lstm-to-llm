"""Tests for head+tail truncation and the stress-test variants (no model download needed: a fake tokenizer is used)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from aes.models.transformer import encode  # noqa: E402
from aes.stress import stress_variants  # noqa: E402


class FakeTok:
    """Word = token; CLS=-1, SEP=-2."""

    cls_token_id, sep_token_id = -1, -2

    def __call__(self, texts, add_special_tokens=False):
        """Each word becomes one token whose id is the word's length, so expected outputs are easy to write."""
        return {"input_ids": [[len(w) for w in t.split()] for t in texts]}


def test_head_tail_truncation_keeps_start_and_end():
    """Long essays keep their first `head` tokens and their last tokens, and fit the length limit exactly."""
    words = " ".join(["a"] * 5 + ["bb"] * 50 + ["ccc"] * 5)  # after "Prompt 1:" prefix: 2 tokens
    out = encode([words], [1], FakeTok(), max_len=12, head=4)[0]
    assert len(out) == 12
    assert out[0] == -1 and out[-1] == -2
    assert out[1:5] == [6, 2, 1, 1]  # "Prompt", "1:", "a", "a"
    assert out[5:11] == [2, 3, 3, 3, 3, 3]  # tail: 10 - 4 = 6 tokens = one "bb" + five "ccc"


def test_short_text_untouched():
    """Essays that fit are not truncated; only the prompt prefix and special tokens are added."""
    out = encode(["x y"], [3], FakeTok(), max_len=512, head=128)[0]
    assert out == [-1, 6, 2, 1, 1, -2]


def test_stress_variants_only_make_essays_longer():
    """Padding keeps the original essay intact and only adds text after it."""
    v = stress_variants(["short essay."])
    assert v["offtopic"][0].startswith("short essay.") and len(v["offtopic"][0]) > 100
    assert v["repeat"][0] == "short essay. short essay."
