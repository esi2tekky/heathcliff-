"""LLM translation of chapter-split texts via Vertex AI (Gemini).

Two translation modes:
  - Sequential: chapter-by-chapter with checkpoint/resume support.
  - Parallel: splits the book into ~N sub-parts and translates them
    concurrently (~20 parallel API calls), then stitches results back
    into the chapter structure.
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import vertexai
from vertexai.generative_models import GenerativeModel

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FR_TO_EN = (
    "You are a literary translator specializing in French-to-English translation. "
    "Translate the following passage of French literature into English. "
    "Preserve the original tone, emotional register, and narrative style "
    "as faithfully as possible. Do not add commentary, notes, or explanations. "
    "Output only the English translation."
)

SYSTEM_PROMPT_EN_TO_FR = (
    "You are a literary translator specializing in English-to-French translation. "
    "Translate the following passage of English literature into French. "
    "Preserve the original tone, emotional register, and narrative style "
    "as faithfully as possible. Do not add commentary, notes, or explanations. "
    "Output only the French translation."
)


def _system_prompt(book: dict) -> str:
    """Return the appropriate system prompt based on translation direction."""
    if book["direction"] == "en_to_fr":
        return SYSTEM_PROMPT_EN_TO_FR
    return SYSTEM_PROMPT_FR_TO_EN


# ---------------------------------------------------------------------------
# Core translation
# ---------------------------------------------------------------------------

def _translate_chapter(
    model: GenerativeModel,
    text: str,
    system_prompt: str,
) -> dict:
    """Translate a single chapter via Vertex AI Gemini.

    Returns a dict with keys: translated_text, input_tokens, output_tokens.
    """
    response = model.generate_content(
        f"{system_prompt}\n\n{text}",
        generation_config={
            "temperature": config.TRANSLATE_TEMPERATURE,
            "max_output_tokens": config.TRANSLATE_MAX_TOKENS,
        },
    )
    translated_text = response.text
    usage = response.usage_metadata
    return {
        "translated_text": translated_text,
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
    }


def translate_book(book: dict, force: bool = False) -> dict:
    """Translate all chapters of a book, with checkpoint/resume support.

    Parameters
    ----------
    book : dict
        Book entry from config.BOOKS.
    force : bool
        If True, discard any existing progress and retranslate from scratch.

    Returns
    -------
    dict
        The complete translation result (same format as chapter-split JSON,
        with translated text and translation metadata).
    """
    slug = book["slug"]
    orig_lang = config.original_lang(book)
    trans_lang = config.translation_lang(book)

    # Input: chapter-split JSON of the original text
    input_path = config.PROCESSED_DIR / f"{slug}_{orig_lang}.json"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Chapter-split file not found: {input_path}. "
            f"Run chapter splitting first."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    source_chapters = source_data["chapters"]
    n_chapters = len(source_chapters)

    # Output path
    config.TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.TRANSLATIONS_DIR / f"{slug}_llm.json"

    # ----- Load existing checkpoint or initialise --------------------------
    result = None
    if output_path.exists() and not force:
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        logger.info(
            "Resuming %s: %d / %d chapters already translated.",
            slug,
            result["metadata"]["n_chapters_translated"],
            n_chapters,
        )
    if result is None:
        # Pre-allocate chapter slots with None
        result = {
            "title": book["title"],
            "author": book["author"],
            "language": trans_lang,
            "source_language": orig_lang,
            "direction": book["direction"],
            "model": config.GEMINI_MODEL_TRANSLATE,
            "n_chapters": n_chapters,
            "chapters": [None] * n_chapters,
            "metadata": {
                "n_chapters_translated": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            },
        }

    # ----- Translate chapter by chapter ------------------------------------
    system_prompt = _system_prompt(book)
    vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
    model = GenerativeModel(config.GEMINI_MODEL_TRANSLATE)

    for i, src_chapter in enumerate(source_chapters):
        # Skip already-translated chapters (checkpoint resume)
        if result["chapters"][i] is not None:
            logger.debug("Chapter %d already translated, skipping.", i + 1)
            continue

        logger.info(
            "Translating %s — chapter %d / %d ...",
            book["title"], i + 1, n_chapters,
        )

        api_result = _translate_chapter(model, src_chapter["text"], system_prompt)

        translated_text = api_result["translated_text"]
        result["chapters"][i] = {
            "number": src_chapter.get("number", i + 1),
            "title": src_chapter.get("title", ""),
            "text": translated_text,
            "char_count": len(translated_text),
            "word_count": len(translated_text.split()),
            "input_tokens": api_result["input_tokens"],
            "output_tokens": api_result["output_tokens"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Update running totals
        result["metadata"]["n_chapters_translated"] += 1
        result["metadata"]["total_input_tokens"] += api_result["input_tokens"]
        result["metadata"]["total_output_tokens"] += api_result["output_tokens"]

        # Checkpoint: save after every chapter
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(
            "  Chapter %d done — %d input tokens, %d output tokens.",
            i + 1,
            api_result["input_tokens"],
            api_result["output_tokens"],
        )

        # Rate-limit delay (skip after last chapter)
        if i < n_chapters - 1:
            time.sleep(config.RATE_LIMIT_DELAY)

    logger.info(
        "Translation complete for %s: %d chapters, "
        "%d total input tokens, %d total output tokens.",
        book["title"],
        result["metadata"]["n_chapters_translated"],
        result["metadata"]["total_input_tokens"],
        result["metadata"]["total_output_tokens"],
    )
    return result


# ---------------------------------------------------------------------------
# Parallel translation (concurrent API calls)
# ---------------------------------------------------------------------------

def _translate_part(
    model: GenerativeModel,
    part_index: int,
    text: str,
    system_prompt: str,
) -> tuple[int, dict]:
    """Translate a single sub-part, returning (index, result_dict).

    Thread-safe wrapper around _translate_chapter for use with
    ThreadPoolExecutor.
    """
    result = _translate_chapter(model, text, system_prompt)
    return part_index, result


def translate_book_parallel(book: dict, force: bool = False) -> dict:
    """Translate a book using ~N parallel Gemini API calls.

    Splits the source text into sub-parts (one per chapter or more), fires
    off up to config.PARALLEL_WORKERS concurrent requests, and stitches the
    results back together in order.

    Parameters
    ----------
    book : dict
        Book entry from config.BOOKS.
    force : bool
        If True, discard any existing progress and retranslate from scratch.

    Returns
    -------
    dict
        Same format as translate_book().
    """
    slug = book["slug"]
    orig_lang = config.original_lang(book)
    trans_lang = config.translation_lang(book)

    input_path = config.PROCESSED_DIR / f"{slug}_{orig_lang}.json"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Chapter-split file not found: {input_path}. "
            f"Run chapter splitting first."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    source_chapters = source_data["chapters"]
    n_chapters = len(source_chapters)

    config.TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.TRANSLATIONS_DIR / f"{slug}_llm.json"

    # ----- Load existing checkpoint or initialise --------------------------
    result = None
    if output_path.exists() and not force:
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        logger.info(
            "Resuming parallel %s: %d / %d chapters already translated.",
            slug,
            result["metadata"]["n_chapters_translated"],
            n_chapters,
        )
    if result is None:
        result = {
            "title": book["title"],
            "author": book["author"],
            "language": trans_lang,
            "source_language": orig_lang,
            "direction": book["direction"],
            "model": config.GEMINI_MODEL_TRANSLATE,
            "n_chapters": n_chapters,
            "chapters": [None] * n_chapters,
            "metadata": {
                "n_chapters_translated": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            },
        }

    # Identify chapters that still need translation.
    pending = [
        (i, src_ch)
        for i, src_ch in enumerate(source_chapters)
        if result["chapters"][i] is None
    ]

    if not pending:
        logger.info("All %d chapters already translated for %s.", n_chapters, slug)
        return result

    logger.info(
        "Parallel translation of %s: %d chapters to translate with up to %d workers.",
        book["title"], len(pending), config.PARALLEL_WORKERS,
    )

    system_prompt = _system_prompt(book)
    vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
    model = GenerativeModel(config.GEMINI_MODEL_TRANSLATE)

    # Fire off all pending chapters concurrently.
    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                _translate_part, model, i, src_ch["text"], system_prompt,
            ): (i, src_ch)
            for i, src_ch in pending
        }

        for future in as_completed(futures):
            i, src_ch = futures[future]
            try:
                _, api_result = future.result()
            except Exception:
                logger.exception(
                    "Failed to translate chapter %d of %s", i + 1, book["title"],
                )
                continue

            translated_text = api_result["translated_text"]
            result["chapters"][i] = {
                "number": src_ch.get("number", i + 1),
                "title": src_ch.get("title", ""),
                "text": translated_text,
                "char_count": len(translated_text),
                "word_count": len(translated_text.split()),
                "input_tokens": api_result["input_tokens"],
                "output_tokens": api_result["output_tokens"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            result["metadata"]["n_chapters_translated"] += 1
            result["metadata"]["total_input_tokens"] += api_result["input_tokens"]
            result["metadata"]["total_output_tokens"] += api_result["output_tokens"]

            # Checkpoint after each completed chapter.
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(
                "  Chapter %d / %d done — %d input tokens, %d output tokens.",
                i + 1, n_chapters,
                api_result["input_tokens"],
                api_result["output_tokens"],
            )

    logger.info(
        "Parallel translation complete for %s: %d chapters, "
        "%d total input tokens, %d total output tokens.",
        book["title"],
        result["metadata"]["n_chapters_translated"],
        result["metadata"]["total_input_tokens"],
        result["metadata"]["total_output_tokens"],
    )
    return result


# ---------------------------------------------------------------------------
# Batch translation
# ---------------------------------------------------------------------------

def translate_all_books(force: bool = False) -> list[dict]:
    """Translate all books defined in config.BOOKS.

    Parameters
    ----------
    force : bool
        If True, retranslate from scratch even if checkpoint files exist.

    Returns
    -------
    list[dict]
        List of translation results, one per book.
    """
    results = []
    for book in config.BOOKS:
        logger.info("=" * 60)
        logger.info("Starting translation: %s", book["title"])
        logger.info("=" * 60)
        result = translate_book(book, force=force)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------

def _dry_run_book(book: dict, force: bool = False) -> None:
    """Print what would be translated without calling the API."""
    slug = book["slug"]
    orig_lang = config.original_lang(book)
    trans_lang = config.translation_lang(book)
    direction_label = f"{orig_lang.upper()} -> {trans_lang.upper()}"

    input_path = config.PROCESSED_DIR / f"{slug}_{orig_lang}.json"
    output_path = config.TRANSLATIONS_DIR / f"{slug}_llm.json"

    print(f"\n{'=' * 60}")
    print(f"  {book['title']} ({book['author']})")
    print(f"  Direction: {direction_label}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    if not input_path.exists():
        print(f"  STATUS: SKIP — input file not found")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    n_chapters = len(source_data["chapters"])
    total_chars = sum(ch.get("char_count", len(ch.get("text", "")))
                      for ch in source_data["chapters"])

    # Check for existing checkpoint
    already_done = 0
    if output_path.exists() and not force:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        already_done = sum(1 for ch in existing.get("chapters", [])
                           if ch is not None)

    remaining = n_chapters - already_done
    print(f"  Chapters: {n_chapters} total, {already_done} already done, "
          f"{remaining} remaining")
    print(f"  Source characters: {total_chars:,}")
    print(f"  Model: {config.GEMINI_MODEL_TRANSLATE}")
    print(f"  Temperature: {config.TRANSLATE_TEMPERATURE}")
    print(f"  Max tokens: {config.TRANSLATE_MAX_TOKENS}")

    if remaining == 0:
        print(f"  STATUS: COMPLETE — all chapters already translated")
    elif force:
        print(f"  STATUS: WOULD RETRANSLATE all {n_chapters} chapters (--force)")
    else:
        print(f"  STATUS: WOULD TRANSLATE {remaining} chapters")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for translation."""
    parser = argparse.ArgumentParser(
        description="Translate chapter-split texts via Vertex AI (Gemini)."
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Title or slug of a single book to translate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="translate_all",
        help="Translate all books in the corpus.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard existing progress and retranslate from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print what would be translated without calling the API.",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.book and not args.translate_all:
        parser.error("Specify --book TITLE or --all.")

    if args.dry_run:
        if args.translate_all:
            for book in config.BOOKS:
                _dry_run_book(book, force=args.force)
        else:
            book = config.get_book(args.book)
            _dry_run_book(book, force=args.force)
        return

    if args.translate_all:
        translate_all_books(force=args.force)
    else:
        book = config.get_book(args.book)
        translate_book(book, force=args.force)


if __name__ == "__main__":
    main()
