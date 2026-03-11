"""Qualitative divergence analysis for the emotional arc pipeline.

Identifies the passages where emotional arcs diverge most between versions
and produces a side-by-side comparison for the milestone writeup.

Usage::

    python -m src.qualitative --book wuthering_heights --methods sw_xlm_roberta
    python -m src.qualitative --all --methods sw_xlm_roberta sw_labmt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import config
from src.sentiment import _load_full_text, _version_language, METHOD_LANGUAGES

logger = logging.getLogger(__name__)

# Maximum characters of passage text to include in excerpts.
_EXCERPT_CHARS = 500


# ---------------------------------------------------------------------------
# Score loading (mirrors metrics._load_scores)
# ---------------------------------------------------------------------------

def _load_scores(slug: str, version: str, method: str) -> np.ndarray | None:
    """Load sentiment scores from processed JSON."""
    path = config.PROCESSED_DIR / f"{slug}_{version}_{method}.json"
    if not path.exists():
        logger.warning("Score file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    scores = data.get("scores") if isinstance(data, dict) else None
    if scores is None:
        return None
    return np.asarray(scores, dtype=float)


def _load_window_meta(slug: str, version: str, method: str) -> dict | None:
    """Load sliding-window metadata (total_words, window_words, n_windows)."""
    path = config.PROCESSED_DIR / f"{slug}_{version}_{method}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "total_words" in data:
        return data
    return None


# ---------------------------------------------------------------------------
# Passage extraction
# ---------------------------------------------------------------------------

def _extract_window_passage(
    full_text: str, window_idx: int, total_words: int,
    window_words: int, n_windows: int,
) -> str:
    """Extract the text passage corresponding to a sliding-window index."""
    words = full_text.split()
    N = len(words)
    W = min(window_words, N)

    if N <= W:
        return full_text[:_EXCERPT_CHARS]

    gap = (N - W) / n_windows
    start = int(round(window_idx * gap))
    end = min(start + W, N)
    passage = " ".join(words[start:end])
    return passage[:_EXCERPT_CHARS]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def find_divergent_windows(
    book: dict,
    version_a: str,
    version_b: str,
    method: str,
    n_top: int = 3,
) -> list[dict]:
    """Find the top-N most divergent windows between two versions.

    Parameters
    ----------
    book:
        Book dict from ``config.BOOKS``.
    version_a, version_b:
        Version labels (e.g. ``'en_original'``, ``'fr_human'``).
    method:
        Sentiment method (e.g. ``'sw_xlm_roberta'``).
    n_top:
        Number of divergent passages to return.

    Returns
    -------
    list[dict]
        Each dict contains window_index, narrative_pct, scores, and
        text excerpts from both versions.
    """
    slug = book["slug"]

    scores_a = _load_scores(slug, version_a, method)
    scores_b = _load_scores(slug, version_b, method)

    if scores_a is None or scores_b is None:
        logger.warning(
            "Missing scores for %s: %s or %s (%s)",
            slug, version_a, version_b, method,
        )
        return []

    # Truncate to equal length.
    n = min(len(scores_a), len(scores_b))
    scores_a = scores_a[:n]
    scores_b = scores_b[:n]

    diff = np.abs(scores_a - scores_b)
    top_indices = np.argsort(-diff)[:n_top]

    # Load full texts for passage extraction.
    try:
        text_a = _load_full_text(book, version_a)
        text_b = _load_full_text(book, version_b)
    except FileNotFoundError as exc:
        logger.warning("Cannot load full text for passage extraction: %s", exc)
        text_a = text_b = ""

    # Get window metadata for sliding-window methods.
    meta_a = _load_window_meta(slug, version_a, method)
    meta_b = _load_window_meta(slug, version_b, method)

    results = []
    for idx in top_indices:
        idx = int(idx)
        pct = (idx / n) * 100 if n > 0 else 0.0

        excerpt_a = ""
        excerpt_b = ""

        if text_a and meta_a:
            excerpt_a = _extract_window_passage(
                text_a, idx,
                meta_a.get("total_words", 0),
                meta_a.get("window_words", config.SW_WINDOW_WORDS),
                meta_a.get("n_windows", n),
            )
        elif text_a:
            # Chapter-level: just show proportional excerpt.
            words = text_a.split()
            start = int(len(words) * idx / n)
            excerpt_a = " ".join(words[start:start + 200])[:_EXCERPT_CHARS]

        if text_b and meta_b:
            excerpt_b = _extract_window_passage(
                text_b, idx,
                meta_b.get("total_words", 0),
                meta_b.get("window_words", config.SW_WINDOW_WORDS),
                meta_b.get("n_windows", n),
            )
        elif text_b:
            words = text_b.split()
            start = int(len(words) * idx / n)
            excerpt_b = " ".join(words[start:start + 200])[:_EXCERPT_CHARS]

        results.append({
            "window_index": idx,
            "narrative_pct": round(pct, 1),
            "score_a": round(float(scores_a[idx]), 4),
            "score_b": round(float(scores_b[idx]), 4),
            "abs_diff": round(float(diff[idx]), 4),
            "version_a": version_a,
            "version_b": version_b,
            "excerpt_a": excerpt_a,
            "excerpt_b": excerpt_b,
        })

    return results


def generate_qualitative_report(
    book: dict,
    method: str,
    n_top: int = 3,
) -> dict:
    """Generate a qualitative divergence report for all version pairs.

    Returns a dict with book metadata and per-pair divergence results.
    """
    versions = config.get_versions(book)
    orig = versions[0]
    human = versions[1]
    llm = versions[2]

    pairs = [
        ("original_vs_human", orig, human),
        ("original_vs_llm", orig, llm),
        ("human_vs_llm", human, llm),
    ]

    report = {
        "title": book["title"],
        "slug": book["slug"],
        "method": method,
        "n_top": n_top,
        "pairs": {},
    }

    for pair_label, va, vb in pairs:
        divergences = find_divergent_windows(book, va, vb, method, n_top=n_top)
        if divergences:
            report["pairs"][pair_label] = divergences

    return report


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_qualitative_json(report: dict, book: dict, method: str) -> Path:
    """Save the report as JSON."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"{book['slug']}_{method}_qualitative.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Qualitative JSON saved to %s", path)
    return path


def save_qualitative_markdown(report: dict, book: dict, method: str) -> Path:
    """Save the report as a Markdown file for the milestone writeup."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"{book['slug']}_{method}_qualitative.md"

    lines: list[str] = []
    lines.append(f"# Qualitative Analysis: {report['title']} ({method})\n")

    pair_titles = {
        "original_vs_human": "Original vs. Human Translation",
        "original_vs_llm": "Original vs. LLM Translation",
        "human_vs_llm": "Human Translation vs. LLM Translation",
    }

    for pair_label, divergences in report.get("pairs", {}).items():
        lines.append(f"\n## {pair_titles.get(pair_label, pair_label)}\n")
        for i, d in enumerate(divergences, 1):
            lines.append(
                f"### Divergence Point {i} "
                f"(Window {d['window_index']}, {d['narrative_pct']}% of narrative)\n"
            )
            lines.append(f"- **{d['version_a']}** score: {d['score_a']}")
            lines.append(f"- **{d['version_b']}** score: {d['score_b']}")
            lines.append(f"- Absolute difference: {d['abs_diff']}\n")

            lines.append(f"**{d['version_a']}:**\n")
            lines.append(f"> {d['excerpt_a']}\n")
            lines.append(f"**{d['version_b']}:**\n")
            lines.append(f"> {d['excerpt_b']}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Qualitative Markdown saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def analyze_all_books_qualitative(
    methods: list[str] | None = None,
    n_top: int = 3,
) -> list[dict]:
    """Run qualitative analysis for every book and method."""
    if methods is None:
        methods = ["sw_xlm_roberta"]

    reports: list[dict] = []
    for book in config.BOOKS:
        for method in methods:
            logger.info("Qualitative analysis: %s / %s", book["slug"], method)
            report = generate_qualitative_report(book, method, n_top=n_top)
            if report.get("pairs"):
                save_qualitative_json(report, book, method)
                save_qualitative_markdown(report, book, method)
                reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Command-line interface for qualitative divergence analysis.

    Examples::

        python -m src.qualitative --book wuthering_heights --methods sw_xlm_roberta
        python -m src.qualitative --all --methods sw_xlm_roberta --n-top 5
    """
    parser = argparse.ArgumentParser(
        description="Find and display the most divergent passages between versions.",
    )
    parser.add_argument(
        "--book", type=str, default=None,
        help="Slug or title of a single book.",
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Analyse every book in the corpus.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["sw_xlm_roberta"],
        help="Sentiment method(s) to use for divergence (default: sw_xlm_roberta).",
    )
    parser.add_argument(
        "--n-top", type=int, default=3,
        help="Number of divergent passages per pair (default: 3).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.book and not args.run_all:
        parser.error("Provide --book SLUG or --all.")

    if args.run_all:
        analyze_all_books_qualitative(methods=args.methods, n_top=args.n_top)
    else:
        book = config.get_book(args.book)
        for method in args.methods:
            report = generate_qualitative_report(book, method, n_top=args.n_top)
            if report.get("pairs"):
                save_qualitative_json(report, book, method)
                save_qualitative_markdown(report, book, method)
                print(f"Report saved for {book['title']} / {method}")
            else:
                print(f"No divergence data available for {book['title']} / {method}")


if __name__ == "__main__":
    main()
