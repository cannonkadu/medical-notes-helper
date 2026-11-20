"""Milestone 4 & Stretch: Interactive CLI for the Medical Notes Helper n-gram predictor."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app import __version__
from app.core import NGramLanguageModel, Suggestion, VALID_SMOOTHING, build_default_model
from app.corpus import load_corpus, tokenize
from app.storage import NotesStorage

console = Console()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_smoothing(default: str = "kneser_ney") -> str:
    raw = os.getenv("DEFAULT_SMOOTHING", default).strip().lower()
    return raw if raw in VALID_SMOOTHING else default


def _order_label(order: int) -> str:
    labels = {1: "unigram", 2: "bigram", 3: "trigram"}
    return labels.get(order, f"{order}-gram")


def _render_suggestions(
    context_text: str,
    suggestions: Sequence[Suggestion],
    smoothing: str,
    plain: bool = False,
) -> None:
    if plain:
        print(f"Context: {context_text!r} (smoothing={smoothing})")
        for idx, s in enumerate(suggestions, start=1):
            flag = " [feedback]" if s.boosted else ""
            print(
                f"  {idx}. {s.word:<18} {s.confidence_pct:6.2f}%  ({_order_label(s.matched_order)}){flag}"
            )
        return

    table = Table(
        title=f"Next-Word Suggestions  [dim](smoothing: {smoothing})[/dim]",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("#", style="bold yellow", width=3, justify="right")
    table.add_column("Suggestion", style="bold white", min_width=16)
    table.add_column("Confidence", justify="right", width=11)
    table.add_column("Signal", width=22)
    table.add_column("Match", style="dim", width=10)

    for idx, s in enumerate(suggestions, start=1):
        bar_len = max(1, min(16, int(round(s.confidence_pct / 6.25))))
        bar = "[green]" + "█" * bar_len + "[/green]" + "[dim]" + "░" * (16 - bar_len) + "[/dim]"
        word_display = f"{s.word} [magenta]*[/magenta]" if s.boosted else s.word
        table.add_row(
            str(idx),
            word_display,
            f"{s.confidence_pct:.2f}%",
            bar,
            _order_label(s.matched_order),
        )

    console.print(table)


def cmd_hello(args: argparse.Namespace) -> int:
    storage = NotesStorage(args.storage)
    model = build_default_model(n=args.order, smoothing=args.smoothing, storage=storage)
    stats = model.stats()

    if args.plain:
        print(f"Medical Notes Helper v{__version__}")
        print(
            f"Loaded {stats['sentences']} anonymized administrative notes "
            f"({stats['vocab_size']} unique tokens, {stats['order']}-gram / {stats['smoothing']})."
        )
        print("Try: python -m app.cli suggest \"the patient arrived\"")
        print("Or:  python -m app.cli draft")
        return 0

    body = (
        f"[bold cyan]Medical Notes Helper[/bold cyan] [dim]v{__version__}[/dim]\n"
        "Next-word predictor for drafting consistent administrative medical notes.\n\n"
        f"[bold]Corpus:[/bold] {stats['sentences']} anonymized notes | "
        f"[bold]Vocabulary:[/bold] {stats['vocab_size']} tokens | "
        f"[bold]Model:[/bold] {stats['order']}-gram ({stats['smoothing']})\n\n"
        "[bold]Quickstart Commands:[/bold]\n"
        '  [green]python -m app.cli suggest "the patient arrived"[/green]\n'
        '  [green]python -m app.cli generate "prior authorization has been" --words 10[/green]\n'
        "  [green]python -m app.cli draft[/green]            [dim]# Interactive drafting studio[/dim]\n"
        "  [green]python -m app.cli evaluate[/green]         [dim]# Compare Laplace vs Add-k vs Kneser-Ney[/dim]"
    )
    console.print(Panel(body, title="Healthcare CLI", border_style="bright_blue", box=box.ROUNDED))
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    storage = NotesStorage(args.storage)
    model = build_default_model(n=args.order, smoothing=args.smoothing, storage=storage)
    context_text = " ".join(args.text).strip()
    boosts = storage.get_feedback_boosts(tokenize(context_text))
    suggestions = model.suggest_next_words(
        context_text,
        top_k=args.top_k,
        smoothing=args.smoothing,
        feedback_boosts=boosts,
    )
    _render_suggestions(context_text, suggestions, smoothing=args.smoothing, plain=args.plain)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    storage = NotesStorage(args.storage)
    model = build_default_model(n=args.order, smoothing=args.smoothing, storage=storage)
    prompt = " ".join(args.prompt).strip()
    completion = model.generate_completion(
        prompt,
        max_words=args.words,
        temperature=args.temperature,
        smoothing=args.smoothing,
        feedback_boosts_fn=storage.get_feedback_boosts,
    )
    if args.plain:
        print(completion)
    else:
        console.print(
            Panel(
                f"[bold white]{completion}[/bold white]",
                title=f"Generated Note ({args.smoothing})",
                border_style="green",
                box=box.ROUNDED,
            )
        )
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    storage = NotesStorage(args.storage)
    accepted = not args.reject
    tokens = tokenize(args.context)
    counts = storage.record_feedback(tokens, args.word, accepted=accepted)
    action = "Accepted" if accepted else "Rejected"
    msg = (
        f"{action} suggestion '{args.word}' for context '{args.context}' "
        f"(accepted={counts['accepted']}, rejected={counts['rejected']})."
    )
    if args.plain:
        print(msg)
    else:
        console.print(f"[bold green]Saved feedback:[/bold green] {msg}")
    return 0


def cmd_add_phrase(args: argparse.Namespace) -> int:
    storage = NotesStorage(args.storage)
    phrase = " ".join(args.phrase).strip()
    added = storage.add_custom_phrase(phrase)
    if args.plain:
        print("Added custom phrase." if added else "Phrase already exists or is empty.")
    else:
        if added:
            console.print(f"[bold green]Added custom clinic phrase:[/bold green] {phrase}")
        else:
            console.print("[yellow]Phrase was empty or already saved.[/yellow]")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    storage = NotesStorage(args.storage)
    model = build_default_model(n=args.order, smoothing=args.smoothing, storage=storage)
    stats = model.stats()
    fb = storage.feedback_summary()
    ngrams = stats["unique_ngrams"]

    if args.plain:
        print(f"Order: {stats['order']}-gram | Smoothing: {stats['smoothing']}")
        print(f"Sentences: {stats['sentences']} | Tokens: {stats['total_tokens']} | Vocab: {stats['vocab_size']}")
        print(f"Unique n-grams: {ngrams}")
        print(f"Feedback: {fb}")
        return 0

    table = Table(title="Model & Corpus Statistics", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("N-gram Order", f"{stats['order']}-gram")
    table.add_row("Default Smoothing", str(stats["smoothing"]))
    table.add_row("Training Sentences", str(stats["sentences"]))
    table.add_row("Total Tokens", str(stats["total_tokens"]))
    table.add_row("Vocabulary Size", str(stats["vocab_size"]))
    for order, count in ngrams.items():
        table.add_row(f"Unique {_order_label(order).capitalize()}s", str(count))
    table.add_row("Custom Clinic Phrases", str(fb["custom_phrases"]))
    table.add_row("Feedback Events (Accept / Reject)", f"{fb['accepted']} / {fb['rejected']}")
    table.add_row("Saved Draft Notes", str(fb["drafts"]))

    console.print(table)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    lines = load_corpus()
    split_idx = max(1, int(len(lines) * 0.85))
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    rows = []
    for mode in VALID_SMOOTHING:
        model = NGramLanguageModel(n=args.order, smoothing=mode)
        model.fit(train_lines)
        ppl = model.perplexity(test_lines)
        rows.append((mode, ppl))

    rows.sort(key=lambda r: r[1])

    if args.plain:
        print(f"Held-out Perplexity ({len(train_lines)} train / {len(test_lines)} test, {args.order}-gram):")
        for mode, ppl in rows:
            print(f"  {mode:<12} {ppl:8.2f}")
        return 0

    table = Table(
        title=f"Smoothing Evaluation ({len(train_lines)} train / {len(test_lines)} held-out notes)",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Smoothing", style="bold")
    table.add_column("Held-out Perplexity (lower is better)", justify="right")
    table.add_column("Rank", justify="center")

    for rank, (mode, ppl) in enumerate(rows, start=1):
        badge = "[bold green]#1 Best[/bold green]" if rank == 1 else f"#{rank}"
        table.add_row(mode, f"{ppl:.2f}", badge)

    console.print(table)
    return 0


def cmd_draft(args: argparse.Namespace) -> int:
    storage = NotesStorage(args.storage)
    model = build_default_model(n=args.order, smoothing=args.smoothing, storage=storage)
    top_k = args.top_k
    smoothing = args.smoothing
    words: List[str] = tokenize(" ".join(args.initial)) if args.initial else []

    if not args.plain:
        console.print(
            Panel(
                "[bold cyan]Interactive Medical Note Drafter[/bold cyan]\n"
                "Type text to append, or enter a suggestion number [bold yellow]1..N[/bold yellow] to accept it.\n"
                "[dim]Commands: :complete  :reject <#>  :k <num>  :smoothing <mode>  :undo  :clear  :save [title]  :quit[/dim]",
                border_style="bright_blue",
                box=box.ROUNDED,
            )
        )
    else:
        print("Interactive Medical Note Drafter (:quit to exit, :help for commands)")

    while True:
        current_text = " ".join(words)
        boosts = storage.get_feedback_boosts(words)
        suggestions = model.suggest_next_words(
            current_text,
            top_k=top_k,
            smoothing=smoothing,
            feedback_boosts=boosts,
        )

        if args.plain:
            print(f"\nNote: {current_text or '(empty)'}")
        else:
            console.print(
                Panel(
                    current_text or "[dim](start typing an administrative medical note...)[/dim]",
                    title="Current Draft",
                    border_style="cyan",
                )
            )

        _render_suggestions(current_text, suggestions, smoothing=smoothing, plain=args.plain)

        try:
            user_input = input("\ndraft> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input in (":q", ":quit", ":exit"):
            break
        if user_input == ":help":
            print(
                "Commands:\n"
                "  <number>            Accept suggestion # and record positive feedback\n"
                "  <text>              Append free text to the current draft\n"
                "  :reject <number>    Downvote suggestion # for the current context\n"
                "  :complete           Auto-complete the rest of the sentence\n"
                "  :k <number>         Change number of suggestions displayed\n"
                "  :smoothing <mode>   Switch smoothing (kneser_ney, laplace, add_k)\n"
                "  :undo               Remove the last word\n"
                "  :clear              Clear the current draft\n"
                "  :save [title]       Save the current draft note\n"
                "  :quit               Exit interactive drafter"
            )
            continue
        if user_input == ":undo":
            if words:
                words.pop()
            continue
        if user_input == ":clear":
            words.clear()
            continue
        if user_input == ":complete":
            completed = model.generate_completion(
                current_text,
                max_words=12,
                smoothing=smoothing,
                feedback_boosts_fn=storage.get_feedback_boosts,
            )
            words = tokenize(completed)
            continue
        if user_input.startswith(":k "):
            try:
                new_k = int(user_input.split(maxsplit=1)[1])
                if new_k >= 1:
                    top_k = new_k
                    print(f"Suggestion count set to {top_k}.")
                else:
                    print("Please choose a number >= 1.")
            except ValueError:
                print("Usage: :k <number>")
            continue
        if user_input.startswith(":smoothing "):
            new_mode = user_input.split(maxsplit=1)[1].strip().lower()
            if new_mode in VALID_SMOOTHING:
                smoothing = new_mode
                print(f"Smoothing switched to '{smoothing}'.")
            else:
                print(f"Valid smoothing modes: {', '.join(VALID_SMOOTHING)}")
            continue
        if user_input.startswith(":reject "):
            try:
                idx = int(user_input.split(maxsplit=1)[1]) - 1
                if 0 <= idx < len(suggestions):
                    chosen_word = suggestions[idx].word
                    storage.record_feedback(words, chosen_word, accepted=False)
                    print(f"Recorded negative feedback for '{chosen_word}'.")
                else:
                    print("Suggestion number out of range.")
            except ValueError:
                print("Usage: :reject <number>")
            continue
        if user_input.startswith(":save"):
            title = user_input[5:].strip() or "Administrative Note"
            if not words:
                print("Draft is empty; nothing to save.")
            else:
                saved = storage.save_draft(title, " ".join(words))
                print(f"Saved draft '{saved['title']}' ({saved['saved_at']}).")
            continue

        # Accept numbered suggestion
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(suggestions):
                chosen_word = suggestions[idx].word
                storage.record_feedback(words, chosen_word, accepted=True)
                words.append(chosen_word)
                continue
            print(f"Enter a number between 1 and {len(suggestions)}.")
            continue

        # Otherwise append typed words
        words.extend(tokenize(user_input))

    if words:
        print(f"\nFinal draft:\n{' '.join(words)}")
    return 0


def _add_common_flags(subparser: argparse.ArgumentParser) -> None:
    """Allow common flags to be passed either before or after the subcommand."""
    subparser.add_argument("--order", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    subparser.add_argument(
        "--smoothing", choices=VALID_SMOOTHING, default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    subparser.add_argument("--storage", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    subparser.add_argument("--plain", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    default_order = _env_int("NGRAM_ORDER", 3)
    default_smoothing = _env_smoothing("kneser_ney")
    default_top_k = _env_int("DEFAULT_TOP_K", 5)

    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Medical Notes Helper — N-gram next-word predictor CLI for administrative medical notes.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=default_order,
        help=f"N-gram order (default: {default_order})",
    )
    parser.add_argument(
        "--smoothing",
        choices=VALID_SMOOTHING,
        default=default_smoothing,
        help=f"Smoothing method (default: {default_smoothing})",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional path to local JSON storage for feedback and custom phrases.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable ANSI/Rich formatting for plain-text output.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # hello
    p_hello = subparsers.add_parser("hello", help="Show welcome banner and model status.")
    _add_common_flags(p_hello)
    p_hello.set_defaults(func=cmd_hello)

    # suggest
    p_suggest = subparsers.add_parser("suggest", help="Suggest next words for an input phrase.")
    p_suggest.add_argument("text", nargs="+", help="Context text to predict the next word for.")
    p_suggest.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=default_top_k,
        help=f"Number of next-word suggestions to display (default: {default_top_k}).",
    )
    _add_common_flags(p_suggest)
    p_suggest.set_defaults(func=cmd_suggest)

    # generate
    p_gen = subparsers.add_parser("generate", help="Auto-complete an administrative note phrase.")
    p_gen.add_argument("prompt", nargs="+", help="Starting text for completion.")
    p_gen.add_argument("--words", type=int, default=12, help="Maximum words to generate (default: 12).")
    p_gen.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic greedy decoding).",
    )
    _add_common_flags(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    # draft
    p_draft = subparsers.add_parser("draft", help="Launch interactive note drafting mode.")
    p_draft.add_argument("initial", nargs="*", help="Optional starting words for the draft.")
    p_draft.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=default_top_k,
        help=f"Initial number of next-word suggestions (default: {default_top_k}).",
    )
    _add_common_flags(p_draft)
    p_draft.set_defaults(func=cmd_draft)

    # feedback
    p_fb = subparsers.add_parser("feedback", help="Record positive or negative feedback on a suggestion.")
    p_fb.add_argument("--context", required=True, help="Input context phrase.")
    p_fb.add_argument("--word", required=True, help="Suggested word to rate.")
    group = p_fb.add_mutually_exclusive_group()
    group.add_argument("--accept", action="store_true", help="Mark suggestion as helpful (default).")
    group.add_argument("--reject", action="store_true", help="Mark suggestion as unhelpful.")
    _add_common_flags(p_fb)
    p_fb.set_defaults(func=cmd_feedback)

    # add-phrase
    p_add = subparsers.add_parser("add-phrase", help="Add a custom clinic phrase to personalize suggestions.")
    p_add.add_argument("phrase", nargs="+", help="Administrative phrase to add.")
    _add_common_flags(p_add)
    p_add.set_defaults(func=cmd_add_phrase)

    # stats
    p_stats = subparsers.add_parser("stats", help="Display corpus, n-gram model, and feedback stats.")
    _add_common_flags(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Benchmark smoothing methods by held-out perplexity.")
    _add_common_flags(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        return cmd_hello(
            argparse.Namespace(
                order=args.order,
                smoothing=args.smoothing,
                storage=args.storage,
                plain=args.plain,
            )
        )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
