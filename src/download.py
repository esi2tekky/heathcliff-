"""Download and cache Project Gutenberg texts for the corpus.

Two-level cache under ``data/raw/``:

* ``{book_id}_raw.txt`` -- the original download, byte-for-byte.
* ``{book_id}.txt``     -- the cleaned body (Gutenberg boilerplate stripped).

Re-running a download is a no-op when both cache files already exist.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

from src import config

GUTENBERG_TXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# Encodings to try, in order.  Many French-language texts on Gutenberg are
# served as ISO-8859-1 rather than UTF-8; a few older uploads use Windows-1252.
_ENCODING_FALLBACK_CHAIN = ("utf-8", "iso-8859-1", "windows-1252")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_with_encoding_fallback(url: str) -> str:
    """Fetch *url* and decode the response body, trying several encodings.

    The function walks through ``_ENCODING_FALLBACK_CHAIN`` in order and
    returns the first successful decode.  If every encoding raises, the
    last ``UnicodeDecodeError`` is propagated.

    Parameters
    ----------
    url:
        Fully-qualified URL to download.

    Returns
    -------
    str
        The decoded text content.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    raw_bytes = response.content
    last_error: UnicodeDecodeError | None = None

    for encoding in _ENCODING_FALLBACK_CHAIN:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    # All encodings failed -- re-raise the last error so callers see a
    # meaningful traceback.
    raise last_error  # type: ignore[misc]


def _strip_gutenberg_markup(text: str) -> str:
    """Extract the literary body from a Project Gutenberg plain-text file.

    Gutenberg files are wrapped with legal boilerplate delimited by lines
    that contain ``*** START OF`` and ``*** END OF`` (case varies across
    editions).  This function performs a **case-insensitive** search for
    those markers and returns only the text between them, with leading /
    trailing whitespace trimmed.

    If the markers are not found the full text is returned unchanged so
    that processing can continue (a warning is printed to stderr).

    Parameters
    ----------
    text:
        The full Gutenberg plain-text file content.

    Returns
    -------
    str
        The stripped body text.
    """
    # Compile case-insensitive patterns for the marker lines.
    start_pattern = re.compile(r"^\*\*\*\s*START OF", re.IGNORECASE | re.MULTILINE)
    end_pattern = re.compile(r"^\*\*\*\s*END OF", re.IGNORECASE | re.MULTILINE)

    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)

    if start_match is None or end_match is None:
        print(
            "[WARNING] Gutenberg START/END markers not found; returning full text.",
            file=sys.stderr,
        )
        return text

    # The body begins on the line *after* the START marker line.
    body_start = text.index("\n", start_match.end()) + 1
    body_end = end_match.start()

    return text[body_start:body_end].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_gutenberg(book_id: int, cache_dir: Path | None = None) -> str:
    """Download a single Gutenberg text, returning the cleaned body.

    Results are cached in *cache_dir* (defaults to ``config.RAW_DIR``):

    * ``{book_id}_raw.txt`` -- verbatim download.
    * ``{book_id}.txt``     -- body with Gutenberg boilerplate removed.

    If both files exist the network is never hit.

    Parameters
    ----------
    book_id:
        Numeric Gutenberg book identifier (e.g. ``4650`` for *Candide*).
    cache_dir:
        Directory to store cached files.  Created automatically if it does
        not exist.

    Returns
    -------
    str
        The cleaned (markup-stripped) text of the book.
    """
    if cache_dir is None:
        cache_dir = config.RAW_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_path = cache_dir / f"{book_id}_raw.txt"
    clean_path = cache_dir / f"{book_id}.txt"

    # Fast path: both cache layers present.
    if raw_path.exists() and clean_path.exists():
        print(f"[cache hit] {clean_path}")
        return clean_path.read_text(encoding="utf-8")

    # --------------- layer 1: raw download ---------------
    if raw_path.exists():
        print(f"[cache hit] {raw_path} (raw)")
        raw_text = raw_path.read_text(encoding="utf-8")
    else:
        url = GUTENBERG_TXT_URL.format(id=book_id)
        print(f"[download]  {url}")
        raw_text = _download_with_encoding_fallback(url)
        raw_path.write_text(raw_text, encoding="utf-8")

    # --------------- layer 2: cleaned text ---------------
    cleaned = _strip_gutenberg_markup(raw_text)
    clean_path.write_text(cleaned, encoding="utf-8")
    print(f"[cleaned]   {clean_path}  ({len(cleaned):,} chars)")

    return cleaned


def download_all_books() -> dict[int, str]:
    """Download every Gutenberg text referenced in ``config.BOOKS``.

    Iterates over all books and downloads both the French (``fr_ids``) and
    English (``en_id``) editions.

    Returns
    -------
    dict[int, str]
        Mapping of Gutenberg book ID to cleaned text body.
    """
    results: dict[int, str] = {}

    for book in config.BOOKS:
        # French edition(s) -- note that fr_ids is a list.
        for fr_id in book["fr_ids"]:
            text = download_gutenberg(fr_id)
            results[fr_id] = text
            print(f"  -> {book['title']} (FR {fr_id}): {text[:200]!r}\n")

        # English edition.
        en_id = book["en_id"]
        text = download_gutenberg(en_id)
        results[en_id] = text
        print(f"  -> {book['title']} (EN {en_id}): {text[:200]!r}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Command-line interface for downloading Gutenberg texts.

    Examples::

        python -m src.download --all
        python -m src.download --book-ids 4650 768
    """
    parser = argparse.ArgumentParser(
        description="Download Project Gutenberg texts for the corpus.",
    )
    parser.add_argument(
        "--book-ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="One or more Gutenberg book IDs to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download every book defined in config.BOOKS.",
    )
    args = parser.parse_args(argv)

    if not args.book_ids and not args.download_all:
        parser.error("Provide --book-ids or --all.")

    if args.download_all:
        texts = download_all_books()
        print(f"\nDownloaded {len(texts)} texts total.")
    else:
        for book_id in args.book_ids:
            text = download_gutenberg(book_id)
            print(f"  -> Book {book_id}: {text[:200]!r}\n")


if __name__ == "__main__":
    main()
