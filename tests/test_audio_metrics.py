"""Tests for the speech recognition measures."""

from core.audio.metrics import normalize_words, word_error_rate, word_precision, word_recall


def test_normalize_words_drops_punctuation_and_case():
    assert normalize_words("Red, I saw Blue's body!") == ["red", "i", "saw", "blue's", "body"]


def test_word_error_rate_counts_substitutions_insertions_and_deletions():
    assert word_error_rate("a b c d", "a b c d") == 0.0
    assert word_error_rate("a b c d", "a x c d") == 0.25
    assert word_error_rate("a b c d", "a b c") == 0.25
    assert word_error_rate("a b c d", "a b c d e") == 0.25


def test_word_error_rate_with_an_empty_reference():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "word") == 1.0


def test_word_recall_matches_words_as_a_multiset():
    assert word_recall("the cat saw the dog", "the dog saw a cat") == 0.8
    assert word_recall("the cat saw the dog", "the the cat saw dog") == 1.0


def test_word_recall_can_ignore_short_words():
    assert word_recall("red saw blue near storage", "saw storage", min_length=4) == 1 / 3
    assert word_recall("a b", "a b", min_length=4) == 0.0


def test_word_precision():
    assert word_precision("red saw blue", "red saw blue") == 1.0
    assert word_precision("red saw blue", "red saw green yellow") == 0.5
    assert word_precision("red", "") == 0.0
