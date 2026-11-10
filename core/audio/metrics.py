"""Text measures for judging how well speech was recognized."""

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9']+")


def normalize_words(text: str) -> list[str]:
    """Lowercases text and splits it into words without punctuation.

    Args:
        text: Any text.

    Returns:
        The words, keeping apostrophes inside words.
    """
    return _WORD.findall(text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Computes the word error rate with edit distance over words.

    Args:
        reference: The known text.
        hypothesis: The recognized text.

    Returns:
        Edits divided by the number of reference words. With an empty reference
        the result is 0 for an empty hypothesis and 1 otherwise.
    """
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def word_recall(reference: str, hypothesis: str, min_length: int = 1) -> float:
    """Computes the share of reference words that appear in the hypothesis.

    Words are matched as a multiset, so a word the reference uses twice must
    appear twice. Short function words match by accident when several people
    speak at once, so `min_length` can restrict the count to longer words.

    Args:
        reference: The known text.
        hypothesis: The recognized text.
        min_length: Shortest reference word to count.

    Returns:
        A value from 0 to 1, or 0 when the reference has no counted words.
    """
    ref = Counter(w for w in normalize_words(reference) if len(w) >= min_length)
    hyp = Counter(normalize_words(hypothesis))
    total = sum(ref.values())
    return sum((ref & hyp).values()) / total if total else 0.0


def word_precision(reference: str, hypothesis: str) -> float:
    """Computes the share of hypothesis words that appear in the reference.

    Args:
        reference: The known text.
        hypothesis: The recognized text.

    Returns:
        A value from 0 to 1, or 0 for an empty hypothesis.
    """
    ref = Counter(normalize_words(reference))
    hyp = Counter(normalize_words(hypothesis))
    total = sum(hyp.values())
    return sum((ref & hyp).values()) / total if total else 0.0
