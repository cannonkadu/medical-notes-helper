"""Milestone 1: Anonymized administrative medical corpus curation, cleaning, and tokenization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

START_TOKEN = "<s>"
END_TOKEN = "</s>"

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "medical_corpus.txt"

# Patterns for PHI anonymization when ingesting raw administrative notes
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
_MRN_RE = re.compile(r"\b(?:MRN|ID|ACCT|ACCOUNT)[\s:#-]*\d{4,}\b", re.IGNORECASE)
_TITLED_NAME_RE = re.compile(
    r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"<[a-z]+>|[a-z0-9]+(?:[-'][a-z0-9]+)*")


def anonymize_text(text: str) -> str:
    """Redact common Protected Health Information (PHI) patterns from administrative notes."""
    result = _EMAIL_RE.sub("<email>", text)
    result = _SSN_RE.sub("<ssn>", result)
    result = _PHONE_RE.sub("<phone>", result)
    result = _DATE_RE.sub("<date>", result)
    result = _MRN_RE.sub("<id>", result)
    result = _TITLED_NAME_RE.sub("<name>", result)
    return result


def clean_text(text: str, anonymize: bool = True) -> str:
    """Normalize text for n-gram modeling: optionally anonymize, lowercase, and tidy whitespace."""
    processed = anonymize_text(text) if anonymize else text
    processed = processed.lower()
    # Keep placeholders, letters, digits, hyphens, apostrophes, and sentence-ending punctuation
    processed = re.sub(r"[^a-z0-9<>'\-\s.!?]", " ", processed)
    processed = re.sub(r"\s+", " ", processed).strip()
    return processed


def tokenize(text: str, anonymize: bool = True) -> List[str]:
    """Clean and split a text snippet into word tokens."""
    cleaned = clean_text(text, anonymize=anonymize)
    return _TOKEN_RE.findall(cleaned)


def split_sentences(text: str) -> List[str]:
    """Split a block of text into non-empty sentence strings."""
    raw_parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [part.strip() for part in raw_parts if part.strip()]


def tokenize_sentences(
    texts: Iterable[str],
    n: int = 3,
    anonymize: bool = True,
) -> List[List[str]]:
    """Convert raw corpus lines into padded token sequences for an n-gram model.

    Prepends (n - 1) start tokens ``<s>`` and appends one end token ``</s>`` per sentence.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    pad_left = [START_TOKEN] * max(n - 1, 1)
    tokenized: List[List[str]] = []

    for entry in texts:
        for sentence in split_sentences(entry):
            tokens = tokenize(sentence, anonymize=anonymize)
            if tokens:
                tokenized.append(pad_left + tokens + [END_TOKEN])

    return tokenized


def load_corpus(path: Path | str | None = None) -> List[str]:
    """Load the anonymized administrative medical text corpus from disk."""
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
