from __future__ import annotations

import logging
import re
from typing import Iterable, List

logger = logging.getLogger(__name__)

try:
    import kss  # type: ignore

    _HAS_KSS = True
except Exception:  # pragma: no cover - module is optional
    kss = None
    _HAS_KSS = False

DEFAULT_MAX_SENTENCE_CHARS = 220
SECONDARY_SPLIT_PATTERN = re.compile(r"(?<=[,;:、·—\-–])\s+")


def split_into_sentences(text: str, max_chars: int = DEFAULT_MAX_SENTENCE_CHARS) -> List[str]:
    """
    Split text into Korean-friendly sentences.

    The function prefers kss if available and falls back to a light-weight regex strategy.
    Afterwards, overly long sentences are further split on secondary delimiters (commas,
    middots, em-dashes, etc.).
    """
    text = (text or "").strip()
    if not text:
        return []

    if _HAS_KSS:
        raw_sentences = kss.split_sentences(text)
    else:
        raw_sentences = _regex_sentence_split(text)

    sentences: List[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentences.extend(split_long_sentence(sentence, max_chars=max_chars))

    return sentences


def split_long_sentence(sentence: str, max_chars: int = DEFAULT_MAX_SENTENCE_CHARS) -> List[str]:
    """
    If the sentence is longer than ``max_chars`` it is split on secondary delimiters.

    A final hard character length split is used as a safety net when punctuation based
    splits are not sufficient (e.g. sentences without commas or whitespace).
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return []

    if len(sentence) <= max_chars:
        return [sentence]

    parts = list(_split_on_secondary(sentence))
    if len(parts) == 1:
        return hard_split_by_length(sentence, max_chars=max_chars)

    merged: List[str] = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue

        candidate = f"{buffer} {part}".strip() if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        if buffer:
            merged.append(buffer)
        if len(part) <= max_chars:
            buffer = part
        else:
            merged.extend(hard_split_by_length(part, max_chars=max_chars))
            buffer = ""

    if buffer:
        merged.append(buffer)

    return merged


def hard_split_by_length(sentence: str, max_chars: int = DEFAULT_MAX_SENTENCE_CHARS) -> List[str]:
    """
    Fallback deterministic split that keeps words together when possible.

    This is used when both the primary sentence splitter and secondary delimiter logic
    cannot produce short enough fragments. The function tries word-wise splitting first
    and then falls back to direct character slicing if necessary.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return []

    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    if not words:
        return _chunk_by_chars(sentence, max_chars)

    fragments: List[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
            continue
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            fragments.append(current)
            current = word

    if current:
        fragments.append(current)

    result: List[str] = []
    for fragment in fragments:
        if len(fragment) <= max_chars:
            result.append(fragment)
        else:
            result.extend(_chunk_by_chars(fragment, max_chars))

    return result


def _regex_sentence_split(text: str) -> List[str]:
    # Normalize newlines to simplify processing.
    normalized = re.sub(r"\r\n?", "\n", text)
    # Insert split markers after sentence terminating punctuation.
    pattern = re.compile(r"(?<=[\.?!。！？])\s+")
    parts: Iterable[str] = pattern.split(normalized)
    sentences: List[str] = []
    for part in parts:
        for sub in re.split(r"\n+", part):
            sub = sub.strip()
            if sub:
                sentences.append(sub)
    return sentences


def _split_on_secondary(sentence: str) -> Iterable[str]:
    if not SECONDARY_SPLIT_PATTERN.search(sentence):
        return [sentence]
    return SECONDARY_SPLIT_PATTERN.split(sentence)


def _chunk_by_chars(text: str, max_chars: int) -> List[str]:
    chunks = []
    for start in range(0, len(text), max_chars):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    if not chunks and text.strip():
        chunks.append(text.strip())
    return chunks
