"""Comprehensive unit tests for corpus processing, n-gram modeling, smoothing, and feedback."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import NGramLanguageModel, build_default_model
from app.corpus import END_TOKEN, START_TOKEN, anonymize_text, clean_text, load_corpus, tokenize, tokenize_sentences
from app.storage import NotesStorage


# --- Milestone 1: Anonymization, Cleaning & Tokenization ---


def test_anonymize_redacts_phi_patterns() -> None:
    raw = (
        "Dr. Jane Smith called 555-234-5678 on 04/12/2026 regarding MRN: 987654, "
        "email jane.doe@clinic.org and SSN 123-45-6789."
    )
    redacted = anonymize_text(raw)
    assert "Jane Smith" not in redacted
    assert "555-234-5678" not in redacted
    assert "04/12/2026" not in redacted
    assert "987654" not in redacted
    assert "jane.doe@clinic.org" not in redacted
    assert "123-45-6789" not in redacted
    assert "<name>" in redacted
    assert "<phone>" in redacted
    assert "<date>" in redacted
    assert "<id>" in redacted
    assert "<email>" in redacted
    assert "<ssn>" in redacted


def test_clean_and_tokenize_preserves_medical_hyphens() -> None:
    cleaned = clean_text("Follow-up visit: HIPAA consent & after-visit summary!!")
    assert "follow-up" in cleaned
    tokens = tokenize("Follow-up visit: HIPAA consent & after-visit summary!!")
    assert tokens == ["follow-up", "visit", "hipaa", "consent", "after-visit", "summary"]


def test_tokenize_sentences_pads_start_and_end_tokens() -> None:
    padded = tokenize_sentences(["Patient checked in. Insurance verified."], n=3)
    assert len(padded) == 2
    assert padded[0][:2] == [START_TOKEN, START_TOKEN]
    assert padded[0][-1] == END_TOKEN
    assert padded[1] == [START_TOKEN, START_TOKEN, "insurance", "verified", END_TOKEN]


def test_default_corpus_loads_and_is_substantial() -> None:
    lines = load_corpus()
    assert len(lines) >= 100
    assert all(isinstance(line, str) and len(line) > 10 for line in lines)


# --- Milestone 2: N-Gram Language Model & Smoothing ---


@pytest.mark.parametrize("smoothing", ["laplace", "add_k", "kneser_ney"])
def test_smoothing_assigns_positive_probability_to_unseen_ngrams(smoothing: str) -> None:
    corpus = [
        "The patient arrived for the follow-up appointment.",
        "Insurance verification was completed prior to the visit.",
        "Please schedule a follow-up visit in four weeks.",
    ]
    model = NGramLanguageModel(n=3, smoothing=smoothing).fit(corpus)
    seen_prob = model.probability("appointment", ("the", "follow-up"))
    unseen_prob = model.probability("insurance", ("the", "follow-up"))

    assert seen_prob > 0.0
    assert unseen_prob > 0.0
    assert seen_prob > unseen_prob


def test_laplace_probabilities_sum_to_one_over_vocab() -> None:
    corpus = [
        "The patient arrived on time.",
        "The patient signed the consent form.",
    ]
    model = NGramLanguageModel(n=2, smoothing="laplace").fit(corpus)
    total = sum(model.probability(word, ("the",)) for word in model.vocab)
    assert total == pytest.approx(1.0, rel=1e-6)


def test_kneser_ney_outperforms_laplace_on_held_out_perplexity() -> None:
    lines = load_corpus()
    split = int(len(lines) * 0.85)
    train, test = lines[:split], lines[split:]

    laplace_model = NGramLanguageModel(n=3, smoothing="laplace").fit(train)
    kn_model = NGramLanguageModel(n=3, smoothing="kneser_ney").fit(train)

    assert kn_model.perplexity(test) < laplace_model.perplexity(test)


# --- Milestone 3: Next-Word Suggestion & Text Generation ---


def test_suggest_next_words_returns_relevant_predictions_and_respects_top_k() -> None:
    model = build_default_model(n=3, smoothing="kneser_ney")
    suggestions = model.suggest_next_words("prior authorization", top_k=4)
    assert len(suggestions) == 4
    words = [s.word for s in suggestions]
    assert any(w in words for w in ("was", "has", "is", "renewal"))
    # Probabilities should be sorted descending
    probs = [s.probability for s in suggestions]
    assert probs == sorted(probs, reverse=True)


def test_generate_completion_extends_prompt() -> None:
    model = build_default_model(n=3, smoothing="kneser_ney")
    result = model.generate_completion("please schedule a follow-up", max_words=8)
    tokens = tokenize(result)
    assert len(tokens) > len(tokenize("please schedule a follow-up"))
    assert "visit" in tokens or "appointment" in tokens


# --- Stretch: Feedback Loop & Custom Phrases Persistence ---


def test_feedback_boosts_accepted_and_penalizes_rejected_suggestions(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path / "feedback.json")
    model = build_default_model(n=3, smoothing="kneser_ney", storage=storage)

    context = "the patient requested"
    baseline = model.suggest_next_words(context, top_k=5)
    top_before = baseline[0].word
    runner_up = baseline[1].word

    # Reject the current top word and accept a runner-up candidate
    for _ in range(3):
        storage.record_feedback(tokenize(context), top_before, accepted=False)
        storage.record_feedback(tokenize(context), runner_up, accepted=True)

    boosts = storage.get_feedback_boosts(tokenize(context))
    boosted = model.suggest_next_words(context, top_k=5, feedback_boosts=boosts)

    assert boosts[runner_up] > 1.0
    assert boosts[top_before] < 1.0
    assert any(s.word == runner_up and s.boosted for s in boosted)
    assert boosted[0].word == runner_up


def test_custom_phrases_enrich_model_vocabulary(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path / "phrases.json")
    storage.add_custom_phrase("Clinic concierge arranged valet parking for the patient.")

    model = build_default_model(n=3, smoothing="kneser_ney", storage=storage)
    assert "concierge" in model.vocab
    suggestions = model.suggest_next_words("clinic concierge", top_k=3)
    assert suggestions[0].word == "arranged"
