"""Tests for the feature extractor, the LSTM tokenizer and the LLM prompt/answer handling."""
import pytest

from aes.models.features import extract_features


def test_features_grow_with_length():
    """Basic sanity of the counting features."""
    f = extract_features(["The cat sat.", "The cat sat on the mat. Then it slept, happily."])
    assert f.n_words[1] > f.n_words[0]
    assert f.n_sentences.tolist() == [1, 2]
    assert f.commas.tolist() == [0, 1]


def test_lstm_tokenizer_matches_nea_conventions():
    """Lowercasing, @PERSON1 -> @person, numbers -> <num>, and the vocabulary size limit."""
    pytest.importorskip("torch")
    from aes.models.lstm import build_vocab, to_ids, tokenize

    assert tokenize("Dear @CAPS1, I've 12 dogs!") == ["dear", "@caps", ",", "i've", "<num>", "dogs", "!"]
    vocab = build_vocab(["a a a b", "a c"], size=5)  # 3 specials + 2 most frequent words
    assert set(vocab) == {"<pad>", "<unk>", "<num>", "a", "b"} or set(vocab) == {"<pad>", "<unk>", "<num>", "a", "c"}
    assert to_ids(["a zzz"], vocab) == [[vocab["a"], 1]]


def test_llm_parser_handles_messy_answers():
    """LLMs do not always return clean JSON; the parser must recover the score or report it as invalid."""
    from aes.models.llm import parse_score

    assert parse_score('{"score": 4, "reason": "ok"}', 1, 6) == 4
    assert parse_score('Sure! ```json\n{"score": 3}\n```', 0, 3) == 3
    assert parse_score("Score: 5 because...", 1, 6) == 5
    assert parse_score('{"score": 9}', 1, 6) is None  # out of range
    assert parse_score("I cannot grade this.", 1, 6) is None
    assert parse_score("", 0, 4) is None


def test_llm_prompt_has_rubric_examples_and_scale():
    """The prompt must contain the rubric, the examples, the essay, and the single-rater scale."""
    from aes.models.llm import build_prompt

    p = build_prompt(8, "RUBRIC TEXT", "my essay", [("ex essay", 20)])
    assert "RUBRIC TEXT" in p and "my essay" in p and "<score>20</score>" in p
    assert "from 0 to 30" in p and "ONE rater" in p
