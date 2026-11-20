"""Milestones 2 & 3: N-gram language model with Laplace, Add-k, and Kneser-Ney smoothing."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.corpus import END_TOKEN, START_TOKEN, load_corpus, tokenize, tokenize_sentences
from app.storage import NotesStorage

VALID_SMOOTHING = ("kneser_ney", "laplace", "add_k")


@dataclass(frozen=True)
class Suggestion:
    """A ranked next-word suggestion produced by the n-gram language model."""

    word: str
    probability: float
    confidence_pct: float
    matched_order: int
    boosted: bool = False


class NGramLanguageModel:
    """From-scratch N-gram language model supporting multiple smoothing methods.

    Supports:
    - ``kneser_ney``: Interpolated Kneser-Ney smoothing with absolute discounting.
    - ``laplace``: Classic add-1 (Laplace) smoothing.
    - ``add_k``: Generalized Lidstone (add-k) smoothing.
    """

    def __init__(
        self,
        n: int = 3,
        smoothing: str = "kneser_ney",
        k: float = 0.1,
        discount: float = 0.75,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if smoothing not in VALID_SMOOTHING:
            raise ValueError(f"Unsupported smoothing '{smoothing}'. Choose from {VALID_SMOOTHING}.")
        if k <= 0:
            raise ValueError("k must be > 0")
        if not (0.0 < discount < 1.0):
            raise ValueError("discount must be between 0 and 1")

        self.n = n
        self.smoothing = smoothing
        self.k = k
        self.discount = discount

        # n-gram counts indexed by order (1..n): ngram_counts[order][tuple_of_tokens] -> int
        self.ngram_counts: Dict[int, Counter[Tuple[str, ...]]] = {
            order: Counter() for order in range(1, n + 1)
        }
        # Context totals for order >= 2: context_counts[order][context_tuple] -> total count
        self.context_counts: Dict[int, Counter[Tuple[str, ...]]] = {
            order: Counter() for order in range(2, n + 1)
        }
        # Unique continuations following a context (for Kneser-Ney lambda weights)
        self.continuations_after: Dict[int, Dict[Tuple[str, ...], Set[str]]] = {
            order: defaultdict(set) for order in range(2, n + 1)
        }
        # Unique predecessors preceding a word (for Kneser-Ney unigram continuation probability)
        self.predecessors_before: Dict[str, Set[str]] = defaultdict(set)
        self.unique_bigrams: int = 0

        self.vocab: Set[str] = set()
        self.sentence_count: int = 0
        self.total_tokens: int = 0
        self._fitted: bool = False

    def fit(self, texts: Iterable[str], anonymize: bool = True) -> "NGramLanguageModel":
        """Train the n-gram model on a collection of text strings."""
        tokenized_sentences = tokenize_sentences(texts, n=self.n, anonymize=anonymize)
        return self.fit_tokenized(tokenized_sentences)

    def fit_tokenized(self, tokenized_sentences: Sequence[Sequence[str]]) -> "NGramLanguageModel":
        """Train the n-gram model on already-tokenized and padded sentences."""
        self.ngram_counts = {order: Counter() for order in range(1, self.n + 1)}
        self.context_counts = {order: Counter() for order in range(2, self.n + 1)}
        self.continuations_after = {order: defaultdict(set) for order in range(2, self.n + 1)}
        self.predecessors_before = defaultdict(set)
        self.vocab = set()
        self.sentence_count = 0
        self.total_tokens = 0

        bigram_types: Set[Tuple[str, str]] = set()

        for tokens in tokenized_sentences:
            if not tokens:
                continue
            self.sentence_count += 1
            for token in tokens:
                if token != START_TOKEN:
                    self.vocab.add(token)
                    self.total_tokens += 1
                self.ngram_counts[1][(token,)] += 1

            length = len(tokens)
            for order in range(2, self.n + 1):
                for idx in range(order - 1, length):
                    ngram = tuple(tokens[idx - order + 1 : idx + 1])
                    ctx = ngram[:-1]
                    word = ngram[-1]
                    self.ngram_counts[order][ngram] += 1
                    self.context_counts[order][ctx] += 1
                    self.continuations_after[order][ctx].add(word)
                    if order == 2:
                        bigram_types.add((ctx[0], word))
                        self.predecessors_before[word].add(ctx[0])

        self.unique_bigrams = len(bigram_types)
        self._fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _ensure_fitted(self) -> None:
        if not self._fitted or not self.vocab:
            raise RuntimeError("Model must be fitted before querying probabilities or suggestions.")

    def _laplace_or_add_k_prob(
        self,
        word: str,
        context: Tuple[str, ...],
        alpha: float,
    ) -> float:
        vocab_sz = max(self.vocab_size, 1)
        order = len(context) + 1

        # Back off to the longest seen context if higher-order context was never observed
        while order > 1:
            ctx = context[-(order - 1) :]
            ctx_total = self.context_counts[order].get(ctx, 0)
            if ctx_total > 0:
                ngram_count = self.ngram_counts[order].get(ctx + (word,), 0)
                return (ngram_count + alpha) / (ctx_total + alpha * vocab_sz)
            order -= 1

        unigram_count = self.ngram_counts[1].get((word,), 0)
        return (unigram_count + alpha) / (self.total_tokens + alpha * vocab_sz)

    def _kneser_ney_prob(self, word: str, context: Tuple[str, ...]) -> float:
        vocab_sz = max(self.vocab_size, 1)
        order = len(context) + 1

        if order <= 1:
            if self.unique_bigrams > 0:
                cont_count = len(self.predecessors_before.get(word, ()))
                # Smooth unigram continuation distribution so unseen OOV words receive
                # calibrated probability mass instead of an extreme log-prob penalty.
                smooth_alpha = 0.25
                return (cont_count + smooth_alpha) / (
                    self.unique_bigrams + smooth_alpha * (vocab_sz + 1)
                )
            unigram_count = self.ngram_counts[1].get((word,), 0)
            return (unigram_count + 1.0) / (self.total_tokens + vocab_sz)

        ctx = context[-(order - 1) :]
        ctx_total = self.context_counts[order].get(ctx, 0)
        lower_prob = self._kneser_ney_prob(word, ctx[1:])

        if ctx_total == 0:
            return lower_prob

        ngram_count = self.ngram_counts[order].get(ctx + (word,), 0)
        discounted = max(ngram_count - self.discount, 0.0) / ctx_total
        num_continuations = len(self.continuations_after[order].get(ctx, ()))
        backoff_weight = (self.discount * num_continuations) / ctx_total
        return discounted + backoff_weight * lower_prob

    def probability(
        self,
        word: str,
        context: Sequence[str] = (),
        smoothing: Optional[str] = None,
    ) -> float:
        """Compute conditional probability P(word | context) under the chosen smoothing."""
        self._ensure_fitted()
        mode = smoothing or self.smoothing
        if mode not in VALID_SMOOTHING:
            raise ValueError(f"Unsupported smoothing '{mode}'. Choose from {VALID_SMOOTHING}.")

        trimmed = tuple(context[-(self.n - 1) :]) if self.n > 1 else ()
        if mode == "laplace":
            return self._laplace_or_add_k_prob(word, trimmed, alpha=1.0)
        if mode == "add_k":
            return self._laplace_or_add_k_prob(word, trimmed, alpha=self.k)
        return self._kneser_ney_prob(word, trimmed)

    def _matched_order(self, word: str, context: Tuple[str, ...]) -> int:
        """Identify the highest n-gram order that directly observed (context, word)."""
        for order in range(min(self.n, len(context) + 1), 1, -1):
            ctx = context[-(order - 1) :]
            if self.ngram_counts[order].get(ctx + (word,), 0) > 0:
                return order
        return 1

    def _prepare_context(self, context_text: str) -> Tuple[str, ...]:
        tokens = tokenize(context_text, anonymize=True)
        if self.n == 1:
            return ()
        needed = self.n - 1
        if len(tokens) < needed:
            padded = [START_TOKEN] * (needed - len(tokens)) + tokens
            return tuple(padded)
        return tuple(tokens[-needed:])

    def suggest_next_words(
        self,
        context_text: str,
        top_k: int = 5,
        smoothing: Optional[str] = None,
        feedback_boosts: Optional[Dict[str, float]] = None,
        include_end: bool = False,
    ) -> List[Suggestion]:
        """Predict the top-k most likely next words for the given input text."""
        self._ensure_fitted()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        context = self._prepare_context(context_text)
        boosts = feedback_boosts or {}

        scored: List[Tuple[str, float, int, bool]] = []
        total_score = 0.0

        for candidate in self.vocab:
            if candidate == START_TOKEN:
                continue
            if candidate == END_TOKEN and not include_end:
                continue

            base_prob = self.probability(candidate, context, smoothing=smoothing)
            multiplier = boosts.get(candidate, 1.0)
            adjusted = base_prob * multiplier
            matched = self._matched_order(candidate, context)
            scored.append((candidate, adjusted, matched, multiplier != 1.0))
            total_score += adjusted

        # Sort by adjusted score descending, then highest matched n-gram order, then alphabetically
        scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
        top_items = scored[:top_k]

        denom = total_score if total_score > 0 else 1.0
        return [
            Suggestion(
                word=word,
                probability=score / denom,
                confidence_pct=round((score / denom) * 100.0, 2),
                matched_order=order,
                boosted=was_boosted,
            )
            for word, score, order, was_boosted in top_items
        ]

    def generate_completion(
        self,
        prompt: str,
        max_words: int = 12,
        temperature: float = 0.0,
        smoothing: Optional[str] = None,
        feedback_boosts_fn: Optional[Callable[[List[str]], Dict[str, float]]] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a continuation of an administrative medical note from a starting prompt."""
        self._ensure_fitted()
        rng = random.Random(seed)
        current_tokens = tokenize(prompt, anonymize=True)
        generated: List[str] = []

        for _ in range(max_words):
            context_str = " ".join(current_tokens)
            boosts = feedback_boosts_fn(current_tokens) if feedback_boosts_fn else None
            suggestions = self.suggest_next_words(
                context_str,
                top_k=8,
                smoothing=smoothing,
                feedback_boosts=boosts,
                include_end=True,
            )
            if not suggestions:
                break

            if temperature <= 0.0:
                chosen = suggestions[0].word
            else:
                weights = [
                    math.pow(max(s.probability, 1e-12), 1.0 / temperature) for s in suggestions
                ]
                chosen = rng.choices([s.word for s in suggestions], weights=weights, k=1)[0]

            if chosen == END_TOKEN:
                break

            generated.append(chosen)
            current_tokens.append(chosen)

        prefix = prompt.strip()
        if not generated:
            return prefix
        if not prefix:
            return " ".join(generated)
        return f"{prefix} {' '.join(generated)}"

    def perplexity(self, texts: Sequence[str], smoothing: Optional[str] = None) -> float:
        """Compute perplexity of the model on a set of held-out sentences (lower is better)."""
        self._ensure_fitted()
        tokenized = tokenize_sentences(texts, n=self.n, anonymize=True)
        log_prob_sum = 0.0
        token_count = 0

        for sentence in tokenized:
            start_idx = max(self.n - 1, 1)
            for idx in range(start_idx, len(sentence)):
                ctx = tuple(sentence[max(0, idx - (self.n - 1)) : idx]) if self.n > 1 else ()
                word = sentence[idx]
                prob = max(self.probability(word, ctx, smoothing=smoothing), 1e-12)
                log_prob_sum += math.log(prob)
                token_count += 1

        if token_count == 0:
            return float("inf")
        return math.exp(-log_prob_sum / token_count)

    def stats(self) -> Dict[str, object]:
        """Return diagnostic statistics for the trained language model."""
        return {
            "order": self.n,
            "smoothing": self.smoothing,
            "sentences": self.sentence_count,
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "unique_ngrams": {
                order: len(self.ngram_counts[order]) for order in range(1, self.n + 1)
            },
        }


def build_default_model(
    n: int = 3,
    smoothing: str = "kneser_ney",
    corpus_path: Path | str | None = None,
    storage: Optional[NotesStorage] = None,
) -> NGramLanguageModel:
    """Load the anonymized medical corpus plus any saved custom phrases and fit a model."""
    corpus_lines = load_corpus(corpus_path)
    if storage is not None:
        corpus_lines.extend(storage.get_custom_phrases())
    model = NGramLanguageModel(n=n, smoothing=smoothing)
    model.fit(corpus_lines, anonymize=True)
    return model
