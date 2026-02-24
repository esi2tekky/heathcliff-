"""Orchestrator for the cross-language emotional arc pipeline.

Runs each phase in sequence: download -> split -> translate ->
sentiment -> metrics -> visualize.  Phases can be selected
individually or run together with ``--phase all``.
"""

import argparse
import logging
import os
import sys

from tqdm import tqdm

from src.download import download_gutenberg, download_all_books
from src.chapter_split import split_book, validate_chapter_counts
from src.translate import translate_book
from src.sentiment import analyze_book, analyze_all_books
from src.metrics import compute_all_book_metrics, save_results_json, save_results_csv
from src.visualize import visualize_all
from src import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ── Helpers ───────────────────────────────────────────────────────────────

PHASE_ORDER = ["download", "split", "translate", "sentiment", "metrics", "visualize"]


def _phase_header(name: str) -> None:
    """Print a prominent phase header."""
    banner = "=" * 60
    print(f"\n{banner}\nPhase: {name.upper()}\n{banner}")


def _filter_books(works: list[str] | None) -> list[dict]:
    """Return the subset of config.BOOKS matching *works*, or all books."""
    if not works:
        return list(config.BOOKS)
    filtered = []
    for w in works:
        try:
            filtered.append(config.get_book(w))
        except ValueError:
            logger.warning("Unknown work %r – skipping", w)
    if not filtered:
        logger.error("No valid works found in --works; aborting")
        sys.exit(1)
    return filtered


# ── Phase runners ─────────────────────────────────────────────────────────

def phase_download(books: list[dict]) -> None:
    """Download raw texts from Project Gutenberg."""
    _phase_header("download")
    try:
        if len(books) == len(config.BOOKS):
            download_all_books()
        else:
            for book in tqdm(books, desc="Downloading"):
                # Download every Gutenberg ID associated with this book
                for gid in book["fr_ids"]:
                    download_gutenberg(gid)
                download_gutenberg(book["en_id"])

        # Preview first 200 characters of each downloaded file
        for book in books:
            for gid in book["fr_ids"]:
                path = config.RAW_DIR / f"{gid}.txt"
                if path.exists():
                    text = path.read_text(encoding="utf-8")
                    print(f"\n[{book['slug']}] {gid}.txt  ({len(text)} chars):\n{text[:200]}")
            en_path = config.RAW_DIR / f"{book['en_id']}.txt"
            if en_path.exists():
                text = en_path.read_text(encoding="utf-8")
                print(f"\n[{book['slug']}] {book['en_id']}.txt  ({len(text)} chars):\n{text[:200]}")
    except Exception:
        logger.exception("Download phase failed")


def phase_split(books: list[dict]) -> None:
    """Split raw texts into chapters for both original and human-translated versions."""
    _phase_header("split")
    try:
        for book in tqdm(books, desc="Splitting"):
            orig_lang = config.original_lang(book)
            trans_lang = config.translation_lang(book)

            # Load and split original-language text
            if orig_lang == "fr":
                gid = book["fr_ids"][0]
            else:
                gid = book["en_id"]
            raw_path = config.RAW_DIR / f"{gid}.txt"
            if raw_path.exists():
                raw_text = raw_path.read_text(encoding="utf-8")
                split_book(book, orig_lang, raw_text)
            else:
                logger.warning("Raw file %s not found – skipping", raw_path)

            # Load and split human-translated text
            if trans_lang == "en":
                gid = book["en_id"]
            else:
                gid = book["fr_ids"][0]
            trans_path = config.RAW_DIR / f"{gid}.txt"
            if trans_path.exists():
                trans_text = trans_path.read_text(encoding="utf-8")
                split_book(book, trans_lang, trans_text)
            else:
                logger.warning("Raw file %s not found – skipping", trans_path)

        # Validate chapter counts across the corpus
        validate_chapter_counts(books)
    except Exception:
        logger.exception("Split phase failed")


def phase_translate(books: list[dict], *, skip_llm: bool = False, force: bool = False) -> None:
    """Translate chapters via the Anthropic API."""
    _phase_header("translate")
    if skip_llm:
        logger.info("--skip-llm-translate flag set; skipping translation phase")
        return
    # Fail-fast: require API key before doing any work
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set. Export it or pass --skip-llm-translate."
        )
        return
    try:
        for book in tqdm(books, desc="Translating"):
            translate_book(book, force=force)
    except Exception:
        logger.exception("Translate phase failed")


def phase_sentiment(
    books: list[dict],
    *,
    methods: list[str],
    force: bool = False,
) -> None:
    """Run sentiment analysis on all chapter versions."""
    _phase_header("sentiment")
    try:
        if len(books) == len(config.BOOKS):
            analyze_all_books(methods=methods, force=force)
        else:
            # analyze_book expects (book, version, method) — iterate explicitly
            from src.sentiment import METHOD_LANGUAGES, _version_language
            for book in tqdm(books, desc="Sentiment"):
                versions = config.get_versions(book)
                for version in versions:
                    lang = _version_language(version)
                    for method in methods:
                        if lang not in METHOD_LANGUAGES[method]:
                            continue
                        try:
                            analyze_book(book, version, method, force=force)
                        except FileNotFoundError as exc:
                            logger.warning("Skipping: %s", exc)
    except Exception:
        logger.exception("Sentiment phase failed")


def phase_metrics(methods: list[str]) -> None:
    """Compute arc-level metrics from sentiment scores."""
    _phase_header("metrics")
    try:
        results = compute_all_book_metrics(methods=methods)
        if results:
            save_results_json(results)
            save_results_csv(results)
        else:
            logger.warning("No metrics results produced")
    except Exception:
        logger.exception("Metrics phase failed")


def phase_visualize(methods: list[str]) -> None:
    """Generate all plots."""
    _phase_header("visualize")
    try:
        visualize_all(methods=methods)
    except Exception:
        logger.exception("Visualize phase failed")


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the cross-language emotional arc analysis pipeline.",
    )
    parser.add_argument(
        "--phase",
        choices=PHASE_ORDER + ["all"],
        default="all",
        help="Pipeline phase to execute (default: all)",
    )
    parser.add_argument(
        "--works",
        nargs="+",
        default=None,
        help="Filter to specific book slugs or titles",
    )
    parser.add_argument(
        "--skip-llm-translate",
        action="store_true",
        help="Skip the LLM translation phase entirely",
    )
    parser.add_argument(
        "--sentiment-methods",
        nargs="+",
        default=["vader", "labmt", "xlm_roberta"],
        help="Sentiment methods to use (default: vader labmt xlm_roberta)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-computation even if outputs already exist",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    books = _filter_books(args.works)
    logger.info(
        "Pipeline starting  phase=%s  books=%s",
        args.phase,
        [b["slug"] for b in books],
    )

    run_all = args.phase == "all"

    # ----- download -----
    if run_all or args.phase == "download":
        phase_download(books)

    # ----- split -----
    if run_all or args.phase == "split":
        phase_split(books)

    # ----- translate -----
    if run_all or args.phase == "translate":
        phase_translate(books, skip_llm=args.skip_llm_translate, force=args.force)

    # ----- sentiment -----
    if run_all or args.phase == "sentiment":
        phase_sentiment(books, methods=args.sentiment_methods, force=args.force)

    # ----- metrics -----
    if run_all or args.phase == "metrics":
        phase_metrics(methods=args.sentiment_methods)

    # ----- visualize -----
    if run_all or args.phase == "visualize":
        phase_visualize(methods=args.sentiment_methods)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
