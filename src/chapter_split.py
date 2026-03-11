"""Chapter boundary detection and book splitting.

Detects chapter headings via regex patterns ordered by specificity,
extracts chapter text/metadata, and writes per-book JSON files to
data/processed/{slug}_{lang}.json.
"""

import re
import json
import logging
import argparse
from pathlib import Path

from src.config import BOOKS, PROCESSED_DIR, RAW_DIR, get_book, original_lang

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chapter-heading patterns, ordered from most specific to least specific.
# Each entry is (label, compiled regex).  The regex is applied with
# re.MULTILINE so ^ anchors to start-of-line.
# ---------------------------------------------------------------------------

_ROMAN = r"[IVXLCDM]+"

PATTERNS_EN = [
    # CHAPTER I / CHAPTER 1 / CHAPTER ONE  (all-caps)
    ("CHAPTER_UPPER",
     re.compile(
         r"^\s*CHAPTER\s+"
         r"(?:" + _ROMAN + r"|\d+|[A-Z][A-Z ]+)"
         r"\s*\.?\s*$",
         re.MULTILINE,
     )),
    # Chapter I / Chapter 1 / Chapter One  (title-case)
    ("Chapter_Title",
     re.compile(
         r"^\s*Chapter\s+"
         r"(?:" + _ROMAN + r"|\d+|[A-Za-z][A-Za-z ]+)"
         r"\s*\.?\s*$",
         re.MULTILINE,
     )),
    # Bare roman numeral on its own line  (e.g.  "  IV  ")
    ("Roman_Bare",
     re.compile(
         r"^\s{0,4}(" + _ROMAN + r")\s*$",
         re.MULTILINE,
     )),
    # BOOK I / BOOK 1
    ("BOOK",
     re.compile(
         r"^\s*BOOK\s+"
         r"(?:" + _ROMAN + r"|\d+)"
         r"\s*\.?\s*$",
         re.MULTILINE,
     )),
]

PATTERNS_FR = [
    # CHAPITRE I / CHAPITRE 1 / CHAPITRE PREMIER  (all-caps)
    ("CHAPITRE_UPPER",
     re.compile(
         r"^\s*CHAPITRE\s+"
         r"(?:" + _ROMAN + r"|\d+|PREMIER|PREMI[ÈE]RE|[A-ZÉÈ][A-ZÉÈ ]+)"
         r"\s*\.?\s*$",
         re.MULTILINE,
     )),
    # Chapitre I / Chapitre 1  (title-case)
    ("Chapitre_Title",
     re.compile(
         r"^\s*Chapitre\s+"
         r"(?:" + _ROMAN + r"|\d+|[A-Za-zéèê][A-Za-zéèê ]+)"
         r"\s*\.?\s*$",
         re.MULTILINE,
     )),
    # Bare roman numeral on its own line
    ("Roman_Bare_FR",
     re.compile(
         r"^\s{0,4}(" + _ROMAN + r")\s*$",
         re.MULTILINE,
     )),
    # PARTIE I / PARTIE 1  (used by some texts; lower priority)
    ("PARTIE",
     re.compile(
         r"^\s*PARTIE\s+"
         r"(?:" + _ROMAN + r"|\d+|PREMI[ÈE]RE)"
         r"\s*\.?\s*$",
         re.MULTILINE,
     )),
    # LIVRE I
    ("LIVRE",
     re.compile(
         r"^\s*LIVRE\s+"
         r"(?:" + _ROMAN + r"|\d+|PREMIER)"
         r"\s*\.?\s*$",
         re.MULTILINE,
     )),
]


def _pattern_list(lang: str):
    """Return the ordered pattern list for the given language code."""
    if lang == "fr":
        return PATTERNS_FR
    return PATTERNS_EN


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_chapter_pattern(text: str, lang: str, expected_chapters: int = None):
    """Try each pattern in order; accept first with plausible match count.

    *Plausible* means:
      - within +-3 of expected_chapters, **or**
      - within +-20 % of expected_chapters.
    If expected_chapters is None every pattern that yields >= 2 matches is
    accepted immediately.

    Returns
    -------
    (pattern_label, compiled_re, list_of_match_positions)
        match_positions are character offsets into *text*.
    None
        if no pattern produces plausible results.
    """
    patterns = _pattern_list(lang)

    for label, regex in patterns:
        all_matches = list(regex.finditer(text))
        # Filter out table-of-contents entries: keep only matches whose
        # next match is > 500 chars away (TOC entries are consecutive lines).
        matches = []
        for i, m in enumerate(all_matches):
            next_start = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(text)
            if next_start - m.start() > 500:
                matches.append(m)
        n = len(matches)
        if n < 2:
            continue

        if expected_chapters is None:
            logger.info("Pattern %s matched %d chapters (no expectation).", label, n)
            return label, regex, [m.start() for m in matches]

        abs_ok = abs(n - expected_chapters) <= 3
        pct_ok = abs(n - expected_chapters) <= 0.20 * expected_chapters
        if abs_ok or pct_ok:
            logger.info(
                "Pattern %s matched %d chapters (expected ~%d).",
                label, n, expected_chapters,
            )
            return label, regex, [m.start() for m in matches]
        else:
            logger.debug(
                "Pattern %s gave %d matches, expected ~%d — skipping.",
                label, n, expected_chapters,
            )

    logger.warning(
        "No pattern produced a plausible chapter count for lang=%s "
        "(expected ~%s).",
        lang, expected_chapters,
    )
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_chapters(text: str, boundaries: list[int]):
    """Build a list of chapter dicts from boundary character positions.

    Each dict contains:
      number      – 1-based chapter index
      heading     – the matched heading line
      title       – heuristic subtitle (first non-empty line after heading
                    if < 100 chars), or empty string
      text        – full chapter body (heading excluded)
      char_count  – len(text)
      word_count  – naive whitespace-split word count
    """
    chapters = []
    lines_cache = text.splitlines(keepends=True)

    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(text)
        raw = text[start:end]

        # Split into lines; first line is the heading.
        raw_lines = raw.split("\n")
        heading = raw_lines[0].strip() if raw_lines else ""

        # Heuristic title: first non-blank line after the heading, if short.
        title = ""
        body_start = 1
        for li, line in enumerate(raw_lines[1:], start=1):
            stripped = line.strip()
            if stripped:
                if len(stripped) < 100:
                    title = stripped
                    body_start = li + 1
                break

        body = "\n".join(raw_lines[body_start:]).strip()
        word_count = len(body.split())

        if word_count < 50:
            logger.warning(
                "Chapter %d (%s) is very short: %d words.",
                idx + 1, heading, word_count,
            )

        chapters.append({
            "number": idx + 1,
            "heading": heading,
            "title": title,
            "text": body,
            "char_count": len(body),
            "word_count": word_count,
        })

    return chapters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_book(book: dict, lang: str, text: str) -> dict:
    """Split *text* into chapters and return a JSON-serialisable dict.

    Parameters
    ----------
    book : dict
        Entry from ``config.BOOKS``.
    lang : str
        Language code (``"en"`` or ``"fr"``).
    text : str
        Full text of the book.

    Returns
    -------
    dict with keys: title, author, language, gutenberg_id, n_chapters,
    chapters (list of chapter dicts).
    """
    expected = book.get("expected_chapters")

    result = _detect_chapter_pattern(text, lang, expected_chapters=expected)
    if result is None:
        raise RuntimeError(
            f"Could not detect chapter boundaries for "
            f"{book['slug']} ({lang}). Expected ~{expected} chapters."
        )

    label, regex, positions = result
    chapters = _extract_chapters(text, positions)

    # Determine the Gutenberg ID for this language.
    if lang == "en":
        gid = book.get("en_id")
    else:
        fr_ids = book.get("fr_ids", [])
        gid = fr_ids[0] if fr_ids else None

    doc = {
        "title": book["title"],
        "author": book["author"],
        "language": lang,
        "gutenberg_id": gid,
        "pattern_used": label,
        "n_chapters": len(chapters),
        "chapters": chapters,
    }

    # Persist to disk.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{book['slug']}_{lang}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote %d chapters to %s", len(chapters), out_path)

    return doc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_chapter_counts(books=None):
    """Print a comparison table of detected chapter counts vs. expected.

    Reads previously-written JSON files from ``PROCESSED_DIR``.
    """
    if books is None:
        books = BOOKS

    header = (
        f"{'Book':<30} {'Expected':>8} {'FR':>6} {'EN':>6} {'Match':>6}"
    )
    print(header)
    print("-" * len(header))

    for book in books:
        expected = book.get("expected_chapters", "?")
        slug = book["slug"]
        counts = {}
        for lang in ("fr", "en"):
            path = PROCESSED_DIR / f"{slug}_{lang}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                counts[lang] = data.get("n_chapters", "?")
            else:
                counts[lang] = "-"

        fr_count = counts.get("fr", "-")
        en_count = counts.get("en", "-")

        # Determine match symbol.
        if isinstance(fr_count, int) and isinstance(en_count, int):
            ok = fr_count == en_count == expected
            symbol = "\u2713" if ok else "\u2717"
        else:
            symbol = "?"

        print(
            f"{book['title']:<30} {str(expected):>8} "
            f"{str(fr_count):>6} {str(en_count):>6} {symbol:>6}"
        )


# ---------------------------------------------------------------------------
# Helpers for loading raw text
# ---------------------------------------------------------------------------

def _load_raw_text(book: dict, lang: str) -> str:
    """Load the cleaned Gutenberg text file from data/raw/{id}.txt.

    The download module stores cleaned text as ``{gutenberg_id}.txt``.
    """
    if lang == "en":
        gid = book["en_id"]
    else:
        gid = book["fr_ids"][0]
    path = RAW_DIR / f"{gid}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw text not found: {path}. "
            f"Run the download step first."
        )
    return path.read_text(encoding="utf-8")


def _process_book(book: dict):
    """Detect chapters in both language versions of a single book."""
    orig = original_lang(book)
    # The other language is whichever one isn't the original.
    trans = "en" if orig == "fr" else "fr"

    for lang in (orig, trans):
        try:
            text = _load_raw_text(book, lang)
        except FileNotFoundError as exc:
            logger.warning("%s — skipping.", exc)
            continue
        try:
            split_book(book, lang, text)
        except RuntimeError as exc:
            logger.error("%s", exc)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for chapter splitting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Split Gutenberg texts into chapters.",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Slug or title of a single book to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="process_all",
        help="Process every book in the corpus.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only print the chapter-count validation table.",
    )
    args = parser.parse_args()

    if args.validate_only:
        validate_chapter_counts()
        return

    if args.book:
        book = get_book(args.book)
        _process_book(book)
    elif args.process_all:
        for book in BOOKS:
            _process_book(book)
    else:
        parser.print_help()
        return

    # Always show validation table after processing.
    print()
    validate_chapter_counts()


if __name__ == "__main__":
    main()
