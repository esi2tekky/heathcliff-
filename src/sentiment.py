"""Multi-method sentiment analysis for the cross-language emotional arc pipeline.

Supports four methods with language-specific applicability:

+----------------+---------+---------+
| Method         | French  | English |
+----------------+---------+---------+
| VADER          |   No    |   Yes   |
| labMT          |   Yes   |   No   |
| XLM-RoBERTa   |   Yes   |   Yes   |
| LLM Judge      |   Yes   |   Yes   |
+----------------+---------+---------+

Usage::

    python -m src.sentiment --all --methods vader labmt xlm_roberta llm_judge
    python -m src.sentiment --book candide --methods vader --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language compatibility map
# ---------------------------------------------------------------------------
METHOD_LANGUAGES: dict[str, set[str]] = {
    "vader": {"en"},
    "labmt": {"fr"},
    "xlm_roberta": {"fr", "en"},
    "llm_judge": {"fr", "en"},
}

ALL_METHODS = list(METHOD_LANGUAGES.keys())

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_xlm_classifier = None


def _get_xlm_classifier():
    """Lazily initialise and return the XLM-RoBERTa sentiment pipeline."""
    global _xlm_classifier
    if _xlm_classifier is None:
        logger.info("Loading XLM-RoBERTa model: %s", config.XLM_MODEL)
        from transformers import pipeline

        _xlm_classifier = pipeline(
            "sentiment-analysis",
            model=config.XLM_MODEL,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("XLM-RoBERTa model loaded.")
    return _xlm_classifier


# ---------------------------------------------------------------------------
# Version -> language helper
# ---------------------------------------------------------------------------

def _version_language(version: str) -> str:
    """Return 'fr' or 'en' based on the version label."""
    if version.startswith("fr"):
        return "fr"
    if version.startswith("en"):
        return "en"
    raise ValueError(f"Cannot determine language for version: {version!r}")


def _version_to_path(book: dict, version: str) -> Path:
    """Map a version label to its chapter-JSON file path.

    Version labels and their corresponding file locations:
      - fr_original / en_original -> data/processed/{slug}_fr.json / _en.json
      - en_human / fr_human       -> data/processed/{slug}_en.json / _fr.json
      - en_llm / fr_llm           -> data/translations/{slug}_llm.json
    """
    slug = book["slug"]
    if version.endswith("_llm"):
        return config.TRANSLATIONS_DIR / f"{slug}_llm.json"
    # original or human: derive language suffix from the version prefix
    lang = _version_language(version)
    return config.PROCESSED_DIR / f"{slug}_{lang}.json"


def _load_chapters(book: dict, version: str) -> list[dict]:
    """Load the chapter JSON for a given book and version.

    Returns a list of chapter dicts, each expected to have at least a
    'text' key (and usually 'chapter_number', 'title', etc.).
    """
    path = _version_to_path(book, version)
    if not path.exists():
        raise FileNotFoundError(
            f"Chapter file not found for {book['title']} / {version}: {path}"
        )
    logger.info("Loading chapters from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both a bare list and a dict with a 'chapters' key.
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "chapters" in data:
        return data["chapters"]
    raise ValueError(f"Unexpected JSON structure in {path}")


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------

def _score_vader(text: str) -> float:
    """Score a single text using VADER. Returns compound score in [-1, 1]."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores["compound"]


# ---------------------------------------------------------------------------
# labMT (French)
# ---------------------------------------------------------------------------

def _load_labmt_lexicon() -> dict[str, float]:
    """Load the French labMT happiness lexicon, caching to disk.

    Downloads from hedonometer.org on first call and stores the result at
    ``config.LABMT_CACHE_PATH``.

    Returns a dict mapping lowercase French words to their happiness scores
    (typically on a 1-9 scale centred around 5).
    """
    cache_path = config.LABMT_CACHE_PATH

    if cache_path.exists():
        logger.info("Loading labMT lexicon from cache: %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Downloading labMT French lexicon from %s", config.LABMT_API_URL)
    import requests

    url = config.LABMT_API_URL
    lexicon: dict[str, float] = {}

    # The hedonometer API is paginated; follow 'next' links.
    while url:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        for obj in payload.get("objects", []):
            word = obj.get("word", "").lower().strip()
            happs = obj.get("happs")
            if word and happs is not None:
                lexicon[word] = float(happs)
        url = payload.get("next")  # None when no more pages

    # Persist cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False)
    logger.info("Cached %d labMT words to %s", len(lexicon), cache_path)

    return lexicon


def _score_labmt(text: str) -> float:
    """Score a French text using the labMT happiness lexicon.

    Tokenises on whitespace/punctuation boundaries, looks up each token in
    the lexicon, and computes ``(mean_happs - 5) / 4`` to produce a value
    in approximately [-1, 1].
    """
    lexicon = _load_labmt_lexicon()
    tokens = re.findall(r"[a-z\u00e0-\u00ff]+", text.lower())
    scores = [lexicon[t] for t in tokens if t in lexicon]
    if not scores:
        logger.warning("labMT: no lexicon matches found; returning 0.0")
        return 0.0
    mean_happs = sum(scores) / len(scores)
    return (mean_happs - 5.0) / 4.0


# ---------------------------------------------------------------------------
# XLM-RoBERTa
# ---------------------------------------------------------------------------

def _xlm_score_single(text: str) -> float:
    """Score a single (short) text with XLM-RoBERTa.

    Maps probabilities: positive -> +1, neutral -> 0, negative -> -1.
    Returns a value in [-1, 1].
    """
    classifier = _get_xlm_classifier()
    results = classifier(text)[0]  # list of {label, score} dicts
    score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    sentiment = 0.0
    for item in results:
        label = item["label"].lower()
        sentiment += score_map.get(label, 0.0) * item["score"]
    return sentiment


def _score_xlm_roberta(text: str) -> float:
    """Score a (potentially long) text with XLM-RoBERTa using chunking.

    Splits the text into ~400-token windows (whitespace-split) with 50-token
    overlap, scores each chunk, then computes a weighted average by chunk
    length (number of tokens).
    """
    words = text.split()
    chunk_size = config.XLM_CHUNK_TOKENS
    overlap = config.XLM_OVERLAP_TOKENS

    # If short enough, score directly.
    if len(words) <= chunk_size:
        return _xlm_score_single(text)

    chunks: list[tuple[str, int]] = []  # (chunk_text, n_tokens)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append((" ".join(chunk_words), len(chunk_words)))
        if end >= len(words):
            break
        start += chunk_size - overlap

    weighted_sum = 0.0
    total_weight = 0
    for chunk_text, n_tokens in chunks:
        score = _xlm_score_single(chunk_text)
        weighted_sum += score * n_tokens
        total_weight += n_tokens

    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# LLM Judge (Anthropic Claude)
# ---------------------------------------------------------------------------

_LLM_JUDGE_PROMPT = (
    "Rate the overall emotional tone of the following passage on a scale "
    "from -10 (extremely negative/dark/despairing) to +10 (extremely "
    "positive/joyful/uplifting). Consider the mood, events, character "
    "emotions, and narrative atmosphere. Respond with ONLY a single number, "
    "nothing else."
)


def _score_llm_judge(text: str) -> float:
    """Score a single text using Claude as an LLM judge.

    Sends the text to ``config.ANTHROPIC_MODEL_JUDGE`` and parses the
    numeric response.  The raw [-10, 10] score is rescaled to [-1, 1].
    """
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL_JUDGE,
        max_tokens=16,
        messages=[
            {
                "role": "user",
                "content": f"{_LLM_JUDGE_PROMPT}\n\n{text}",
            },
        ],
    )
    raw_text = message.content[0].text.strip()
    # Parse the numeric response, allowing decimals and negative signs.
    match = re.search(r"-?\d+(?:\.\d+)?", raw_text)
    if match is None:
        logger.warning("LLM judge returned non-numeric response: %r; defaulting to 0", raw_text)
        return 0.0
    raw_score = float(match.group())
    # Clamp to [-10, 10] then rescale to [-1, 1].
    raw_score = max(-10.0, min(10.0, raw_score))
    return raw_score / 10.0


# ---------------------------------------------------------------------------
# Scoring dispatcher
# ---------------------------------------------------------------------------

_SCORE_FUNCTIONS: dict[str, callable] = {
    "vader": _score_vader,
    "labmt": _score_labmt,
    "xlm_roberta": _score_xlm_roberta,
    "llm_judge": _score_llm_judge,
}


# ---------------------------------------------------------------------------
# Main analysis functions
# ---------------------------------------------------------------------------

def analyze_book(
    book: dict,
    version: str,
    method: str,
    force: bool = False,
) -> dict:
    """Run a single sentiment method on one book+version.

    Parameters
    ----------
    book:
        Book dict from ``config.BOOKS``.
    version:
        Version label (e.g. ``'fr_original'``, ``'en_human'``, ``'en_llm'``).
    method:
        One of ``'vader'``, ``'labmt'``, ``'xlm_roberta'``, ``'llm_judge'``.
    force:
        If True, re-run even if the output file already exists.

    Returns
    -------
    dict
        Contains keys: title, version, method, scores, n_chapters.
    """
    slug = book["slug"]
    output_path = config.PROCESSED_DIR / f"{slug}_{version}_{method}.json"

    # Caching: skip if output already exists.
    if output_path.exists() and not force:
        logger.info("Cached result found, skipping: %s", output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    lang = _version_language(version)
    if lang not in METHOD_LANGUAGES[method]:
        raise ValueError(
            f"Method {method!r} does not support language {lang!r} "
            f"(version={version!r})."
        )

    chapters = _load_chapters(book, version)
    score_fn = _SCORE_FUNCTIONS[method]

    scores: list[float] = []
    for i, chapter in enumerate(chapters):
        text = chapter.get("text", "")
        if not text:
            logger.warning(
                "%s / %s / %s: chapter %d has no text; scoring as 0.0",
                book["title"], version, method, i + 1,
            )
            scores.append(0.0)
            continue
        logger.info(
            "Scoring %s ch.%d/%d [%s/%s]",
            slug, i + 1, len(chapters), version, method,
        )
        score = score_fn(text)
        scores.append(score)

        # Rate-limit LLM judge calls.
        if method == "llm_judge":
            time.sleep(config.RATE_LIMIT_DELAY)

    result = {
        "title": book["title"],
        "version": version,
        "method": method,
        "scores": scores,
        "n_chapters": len(chapters),
    }

    # Persist result.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", output_path)

    return result


def analyze_all_books(
    methods: list[str] | None = None,
    force: bool = False,
) -> list[dict]:
    """Run sentiment analysis across all books, versions, and methods.

    For each book, determines the applicable versions via
    ``config.get_versions()`` and only runs methods compatible with each
    version's language.

    Parameters
    ----------
    methods:
        List of method names to run. Defaults to all four methods.
    force:
        If True, re-run even when cached output exists.

    Returns
    -------
    list[dict]
        A list of result dicts, one per (book, version, method) combination.
    """
    if methods is None:
        methods = ALL_METHODS

    results: list[dict] = []

    for book in config.BOOKS:
        versions = config.get_versions(book)
        logger.info(
            "Processing %s — versions: %s", book["title"], versions,
        )
        for version in versions:
            lang = _version_language(version)
            for method in methods:
                if lang not in METHOD_LANGUAGES[method]:
                    logger.debug(
                        "Skipping %s/%s/%s (language %s not supported)",
                        book["slug"], version, method, lang,
                    )
                    continue
                try:
                    result = analyze_book(book, version, method, force=force)
                    results.append(result)
                except FileNotFoundError as exc:
                    logger.warning("Skipping: %s", exc)
                except Exception:
                    logger.exception(
                        "Error analysing %s / %s / %s",
                        book["title"], version, method,
                    )

    logger.info("Completed %d analyses.", len(results))
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Command-line interface for running sentiment analysis.

    Examples::

        python -m src.sentiment --all --methods vader xlm_roberta
        python -m src.sentiment --book candide --methods labmt --force
    """
    parser = argparse.ArgumentParser(
        description="Run multi-method sentiment analysis on the corpus.",
    )
    parser.add_argument(
        "--book",
        type=str,
        metavar="TITLE_OR_SLUG",
        help="Analyse a single book (by title or slug).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="analyze_all",
        help="Analyse every book in config.BOOKS.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=ALL_METHODS,
        metavar="METHOD",
        help=(
            f"Sentiment methods to run. Choices: {', '.join(ALL_METHODS)}. "
            "Defaults to all."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run analysis even if cached output exists.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.book and not args.analyze_all:
        parser.error("Provide --book TITLE_OR_SLUG or --all.")

    if args.analyze_all:
        analyze_all_books(methods=args.methods, force=args.force)
    else:
        book = config.get_book(args.book)
        versions = config.get_versions(book)
        for version in versions:
            lang = _version_language(version)
            for method in args.methods:
                if lang not in METHOD_LANGUAGES[method]:
                    logger.info(
                        "Skipping %s/%s/%s (language mismatch)",
                        book["slug"], version, method,
                    )
                    continue
                try:
                    analyze_book(book, version, method, force=args.force)
                except FileNotFoundError as exc:
                    logger.warning("Skipping: %s", exc)
                except Exception:
                    logger.exception(
                        "Error analysing %s / %s / %s",
                        book["title"], version, method,
                    )


if __name__ == "__main__":
    main()
