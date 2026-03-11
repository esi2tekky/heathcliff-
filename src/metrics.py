"""Phase 5: Metrics -- DTW, correlation, and critical point analysis.

Compares emotional arcs pairwise across the three versions of each book
(original, human translation, LLM translation) using multiple distance and
similarity metrics.

Outputs:
    - ``output/results.json``      -- full nested metrics dictionary.
    - ``output/results_table.csv`` -- flattened summary for easy inspection.

Usage::

    python -m src.metrics --all
    python -m src.metrics --book candide --methods xlm_roberta vader
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelmax, argrelmin, savgol_filter
from scipy.stats import pearsonr, spearmanr

from src import config

logger = logging.getLogger(__name__)

# Sentinel list of known sentiment methods used across the pipeline.
DEFAULT_METHODS = ["vader", "labmt", "xlm_roberta", "llm_judge", "sw_xlm_roberta", "sw_labmt"]

# Pair labels in consistent order.
PAIR_LABELS = ("original_vs_human", "original_vs_llm", "human_vs_llm")


# ---------------------------------------------------------------------------
# Low-level arc utilities
# ---------------------------------------------------------------------------

def _truncate_arcs(
    arc_a: np.ndarray,
    arc_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate two arcs to the shorter length, logging a warning if needed.

    Parameters
    ----------
    arc_a, arc_b:
        One-dimensional arrays of sentiment scores.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arcs truncated to ``min(len(arc_a), len(arc_b))``.
    """
    len_a, len_b = len(arc_a), len(arc_b)
    if len_a != len_b:
        logger.warning(
            "Arc lengths differ (%d vs %d); truncating to %d.",
            len_a,
            len_b,
            min(len_a, len_b),
        )
        n = min(len_a, len_b)
        arc_a = arc_a[:n]
        arc_b = arc_b[:n]
    return arc_a, arc_b


def resample_arc(arc: np.ndarray, n_windows: int | None = None) -> np.ndarray:
    """Resample *arc* to *n_windows* evenly-spaced points via linear interpolation.

    Parameters
    ----------
    arc:
        Original arc (1-D array).
    n_windows:
        Target number of windows.  Defaults to ``config.N_PERCENTAGE_WINDOWS``.

    Returns
    -------
    np.ndarray
        Resampled arc of length *n_windows*.
    """
    if n_windows is None:
        n_windows = config.N_PERCENTAGE_WINDOWS
    x_old = np.linspace(0, 1, len(arc))
    x_new = np.linspace(0, 1, n_windows)
    return np.interp(x_new, x_old, arc)


# ---------------------------------------------------------------------------
# DTW (from scratch)
# ---------------------------------------------------------------------------

def compute_dtw(arc_a: np.ndarray, arc_b: np.ndarray) -> float:
    """Dynamic Time Warping distance, normalized by warping-path length.

    Uses a standard O(n*m) dynamic-programming formulation with Euclidean
    (absolute-difference) local cost.  The total accumulated cost is divided
    by the number of steps in the optimal warping path so that the metric is
    comparable across arc pairs of different lengths.

    Parameters
    ----------
    arc_a, arc_b:
        One-dimensional sentiment score arrays.

    Returns
    -------
    float
        Path-length-normalized DTW distance.
    """
    n = len(arc_a)
    m = len(arc_b)

    # Cost matrix -- initialize to infinity except (0, 0).
    cost = np.full((n, m), np.inf)
    cost[0, 0] = abs(arc_a[0] - arc_b[0])

    # First column.
    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + abs(arc_a[i] - arc_b[0])

    # First row.
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + abs(arc_a[0] - arc_b[j])

    # Fill the rest of the matrix.
    for i in range(1, n):
        for j in range(1, m):
            local_cost = abs(arc_a[i] - arc_b[j])
            cost[i, j] = local_cost + min(
                cost[i - 1, j],      # insertion
                cost[i, j - 1],      # deletion
                cost[i - 1, j - 1],  # match
            )

    # Trace back the optimal warping path to get its length.
    i, j = n - 1, m - 1
    path_length = 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            argmin = int(np.argmin(candidates))
            if argmin == 0:
                i -= 1
                j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
        path_length += 1

    return float(cost[n - 1, m - 1] / path_length)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(arc_a: np.ndarray, arc_b: np.ndarray) -> float:
    """Cosine similarity between two arc vectors.

    Returns 0.0 if either vector has zero norm (avoiding division by zero).
    """
    dot = np.dot(arc_a, arc_b)
    norm_a = np.linalg.norm(arc_a)
    norm_b = np.linalg.norm(arc_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Smoothing and critical-point helpers
# ---------------------------------------------------------------------------

def smooth_arc(
    arc: np.ndarray,
    window: int | None = None,
    polyorder: int | None = None,
) -> np.ndarray:
    """Smooth an arc with a Savitzky-Golay filter.

    If the arc is too short for the requested *window* the window is
    progressively reduced (keeping it odd) until it fits, or smoothing is
    skipped entirely when fewer than ``polyorder + 2`` points are available.

    Parameters
    ----------
    arc:
        Raw sentiment scores (1-D).
    window:
        Savitzky-Golay window length (must be odd).  Defaults to
        ``config.SMOOTHING_WINDOW``.
    polyorder:
        Polynomial order.  Defaults to ``config.SMOOTHING_POLYORDER``.

    Returns
    -------
    np.ndarray
        Smoothed arc (same length as input).
    """
    if window is None:
        window = config.SMOOTHING_WINDOW
    if polyorder is None:
        polyorder = config.SMOOTHING_POLYORDER

    n = len(arc)

    # Cannot smooth with fewer than polyorder + 2 points.
    if n < polyorder + 2:
        logger.debug(
            "Arc too short (%d points) for smoothing (need >= %d); "
            "returning unsmoothed.",
            n,
            polyorder + 2,
        )
        return arc.copy()

    # Reduce window to fit the data (must be odd and > polyorder).
    while window > n or window <= polyorder:
        window -= 2
    if window < polyorder + 1:
        window = polyorder + 1
    # Ensure window is odd.
    if window % 2 == 0:
        window += 1
    if window > n:
        return arc.copy()

    return savgol_filter(arc, window_length=window, polyorder=polyorder)


def _find_peaks(arc: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of the top-*k* peaks (local maxima) by amplitude.

    Uses ``scipy.signal.argrelmax`` with ``order=1``.  If fewer than *top_k*
    peaks are found the available set is returned.

    Parameters
    ----------
    arc:
        Smoothed 1-D array.
    top_k:
        Maximum number of peaks to return.

    Returns
    -------
    np.ndarray
        Indices of peaks sorted by descending amplitude.
    """
    indices = argrelmax(arc, order=1)[0]
    if len(indices) == 0:
        return indices
    # Sort by amplitude (descending) and keep top-k.
    sorted_idx = indices[np.argsort(-arc[indices])]
    return sorted_idx[:top_k]


def _find_troughs(arc: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of the top-*k* troughs (local minima) by depth.

    Uses ``scipy.signal.argrelmin`` with ``order=1``.  If fewer than *top_k*
    troughs are found the available set is returned.

    Parameters
    ----------
    arc:
        Smoothed 1-D array.
    top_k:
        Maximum number of troughs to return.

    Returns
    -------
    np.ndarray
        Indices of troughs sorted by ascending amplitude (deepest first).
    """
    indices = argrelmin(arc, order=1)[0]
    if len(indices) == 0:
        return indices
    sorted_idx = indices[np.argsort(arc[indices])]
    return sorted_idx[:top_k]


def compute_peak_alignment(
    arc_a: np.ndarray,
    arc_b: np.ndarray,
    top_k: int | None = None,
) -> float:
    """Average positional shift between nearest peaks, as % of arc length.

    For each peak in *arc_a*, the nearest peak in *arc_b* is found and the
    positional difference is expressed as a fraction of the arc length
    (0.0 = perfect alignment, 1.0 = maximum misalignment).

    If either arc has no detectable peaks, ``float('nan')`` is returned.

    Parameters
    ----------
    arc_a, arc_b:
        Raw (unsmoothed) arcs; smoothing is applied internally.
    top_k:
        Number of peaks to consider.  Defaults to ``config.PEAK_TOP_K``.

    Returns
    -------
    float
        Mean positional shift in [0, 1] or ``nan``.
    """
    if top_k is None:
        top_k = config.PEAK_TOP_K

    smoothed_a = smooth_arc(arc_a)
    smoothed_b = smooth_arc(arc_b)
    n = len(smoothed_a)

    peaks_a = _find_peaks(smoothed_a, top_k)
    peaks_b = _find_peaks(smoothed_b, top_k)

    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return float("nan")

    shifts: list[float] = []
    for pa in peaks_a:
        # Distance (in index space) to every peak in b.
        distances = np.abs(peaks_b.astype(float) - float(pa))
        shifts.append(float(np.min(distances)) / n)

    return float(np.mean(shifts))


def compute_trough_alignment(
    arc_a: np.ndarray,
    arc_b: np.ndarray,
    top_k: int | None = None,
) -> float:
    """Average positional shift between nearest troughs, as % of arc length.

    Mirrors :func:`compute_peak_alignment` but uses local minima instead.

    Parameters
    ----------
    arc_a, arc_b:
        Raw (unsmoothed) arcs; smoothing is applied internally.
    top_k:
        Number of troughs to consider.  Defaults to ``config.PEAK_TOP_K``.

    Returns
    -------
    float
        Mean positional shift in [0, 1] or ``nan``.
    """
    if top_k is None:
        top_k = config.PEAK_TOP_K

    smoothed_a = smooth_arc(arc_a)
    smoothed_b = smooth_arc(arc_b)
    n = len(smoothed_a)

    troughs_a = _find_troughs(smoothed_a, top_k)
    troughs_b = _find_troughs(smoothed_b, top_k)

    if len(troughs_a) == 0 or len(troughs_b) == 0:
        return float("nan")

    shifts: list[float] = []
    for ta in troughs_a:
        distances = np.abs(troughs_b.astype(float) - float(ta))
        shifts.append(float(np.min(distances)) / n)

    return float(np.mean(shifts))


# ---------------------------------------------------------------------------
# Main metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(arc_a: np.ndarray, arc_b: np.ndarray) -> dict:
    """Compute the full suite of comparison metrics between two emotional arcs.

    Arcs are truncated to equal length if necessary (with a warning).

    Parameters
    ----------
    arc_a, arc_b:
        One-dimensional NumPy arrays of chapter-level sentiment scores.

    Returns
    -------
    dict
        Keys: ``pearson_r``, ``pearson_p``, ``spearman_rho``, ``spearman_p``,
        ``dtw_distance``, ``rmse``, ``mean_abs_diff``, ``cosine_similarity``,
        ``peak_alignment``, ``trough_alignment``.
    """
    arc_a, arc_b = _truncate_arcs(
        np.asarray(arc_a, dtype=float),
        np.asarray(arc_b, dtype=float),
    )

    # Guard: need at least 3 points for meaningful correlation.
    if len(arc_a) < 3:
        logger.warning("Arcs have fewer than 3 points; correlation metrics "
                        "will be unreliable.")

    pr, pp = pearsonr(arc_a, arc_b)
    sr, sp = spearmanr(arc_a, arc_b)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "dtw_distance": compute_dtw(arc_a, arc_b),
        "rmse": float(np.sqrt(np.mean((arc_a - arc_b) ** 2))),
        "mean_abs_diff": float(np.mean(np.abs(arc_a - arc_b))),
        "cosine_similarity": cosine_similarity(arc_a, arc_b),
        "peak_alignment": compute_peak_alignment(arc_a, arc_b),
        "trough_alignment": compute_trough_alignment(arc_a, arc_b),
    }


# ---------------------------------------------------------------------------
# Loading sentiment scores from processed JSON files
# ---------------------------------------------------------------------------

def _load_scores(slug: str, version: str, method: str) -> np.ndarray | None:
    """Load sentiment scores from ``data/processed/{slug}_{version}_{method}.json``.

    Returns ``None`` (with a warning) if the file does not exist or lacks a
    ``scores`` key.
    """
    path = config.PROCESSED_DIR / f"{slug}_{version}_{method}.json"
    if not path.exists():
        logger.warning("Score file not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    scores = data.get("scores")
    if scores is None:
        logger.warning("No 'scores' key in %s", path)
        return None
    return np.asarray(scores, dtype=float)


# ---------------------------------------------------------------------------
# Book-level and corpus-level metric computation
# ---------------------------------------------------------------------------

def compute_book_metrics(
    book: dict,
    methods: list[str] | None = None,
) -> dict:
    """Compute all pairwise metrics for a single book.

    For each sentiment method the three version-pairs are compared using
    both raw chapter-aligned arcs and percentage-normalized (resampled) arcs.

    Parameters
    ----------
    book:
        A book dictionary from ``config.BOOKS``.
    methods:
        Sentiment method names to evaluate.  Defaults to
        ``DEFAULT_METHODS``.

    Returns
    -------
    dict
        Nested mapping::

            {
                method: {
                    pair_label: {
                        "chapter": { ...metrics... },
                        "pct_normalized": { ...metrics... },
                    }
                }
            }

        where *pair_label* is one of ``original_vs_human``,
        ``original_vs_llm``, or ``human_vs_llm``.
    """
    if methods is None:
        methods = DEFAULT_METHODS

    slug = book["slug"]
    v_original, v_human, v_llm = config.get_versions(book)

    results: dict = {}

    for method in methods:
        scores_orig = _load_scores(slug, v_original, method)
        scores_human = _load_scores(slug, v_human, method)
        scores_llm = _load_scores(slug, v_llm, method)

        if scores_orig is None or scores_human is None or scores_llm is None:
            logger.warning(
                "Skipping method '%s' for '%s' -- missing score files.",
                method,
                book["title"],
            )
            continue

        # Build the three pairs.
        pairs = {
            "original_vs_human": (scores_orig, scores_human),
            "original_vs_llm": (scores_orig, scores_llm),
            "human_vs_llm": (scores_human, scores_llm),
        }

        method_results: dict = {}
        for pair_label, (arc_a, arc_b) in pairs.items():
            # Chapter-aligned metrics (truncate if lengths differ).
            chapter_metrics = compute_all_metrics(arc_a, arc_b)

            # Percentage-normalized metrics.
            arc_a_pct = resample_arc(arc_a)
            arc_b_pct = resample_arc(arc_b)
            pct_metrics = compute_all_metrics(arc_a_pct, arc_b_pct)

            method_results[pair_label] = {
                "chapter": chapter_metrics,
                "pct_normalized": pct_metrics,
            }

        results[method] = method_results

    return results


def compute_all_book_metrics(
    methods: list[str] | None = None,
) -> dict:
    """Run :func:`compute_book_metrics` for every book in the corpus.

    Parameters
    ----------
    methods:
        Sentiment methods to evaluate.  Defaults to ``DEFAULT_METHODS``.

    Returns
    -------
    dict
        Mapping of book slug to its metrics dictionary.
    """
    all_results: dict = {}
    for book in config.BOOKS:
        logger.info("Computing metrics for '%s' ...", book["title"])
        book_results = compute_book_metrics(book, methods=methods)
        if book_results:
            all_results[book["slug"]] = book_results
    return all_results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: object) -> object:
    """Convert numpy/float types so that ``json.dumps`` does not choke."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def save_results_json(results: dict, path: Path | None = None) -> Path:
    """Write the full results dictionary to ``output/results.json``.

    Parameters
    ----------
    results:
        The nested metrics dict returned by :func:`compute_all_book_metrics`.
    path:
        Override output path.  Defaults to ``config.OUTPUT_DIR / "results.json"``.

    Returns
    -------
    Path
        The path the file was written to.
    """
    if path is None:
        path = config.OUTPUT_DIR / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize_for_json(results), fh, indent=2, ensure_ascii=False)
    logger.info("Full results written to %s", path)
    return path


def save_results_csv(results: dict, path: Path | None = None) -> Path:
    """Flatten *results* into a summary CSV at ``output/results_table.csv``.

    Columns: ``book``, ``method``, ``pair``, ``normalization``, and one column
    per metric name.

    Parameters
    ----------
    results:
        The nested metrics dict returned by :func:`compute_all_book_metrics`.
    path:
        Override output path.  Defaults to
        ``config.OUTPUT_DIR / "results_table.csv"``.

    Returns
    -------
    Path
        The path the file was written to.
    """
    if path is None:
        path = config.OUTPUT_DIR / "results_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
        "dtw_distance",
        "rmse",
        "mean_abs_diff",
        "cosine_similarity",
        "peak_alignment",
        "trough_alignment",
    ]

    rows: list[dict] = []
    for book_slug, methods_dict in results.items():
        for method, pairs_dict in methods_dict.items():
            for pair_label, norm_dict in pairs_dict.items():
                for norm_label, metrics in norm_dict.items():
                    row: dict = {
                        "book": book_slug,
                        "method": method,
                        "pair": pair_label,
                        "normalization": norm_label,
                    }
                    for mk in metric_keys:
                        val = metrics.get(mk)
                        row[mk] = val
                    rows.append(row)

    fieldnames = ["book", "method", "pair", "normalization"] + metric_keys

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary CSV written to %s (%d rows)", path, len(rows))
    return path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Command-line interface for computing arc comparison metrics.

    Examples::

        python -m src.metrics --all
        python -m src.metrics --book candide
        python -m src.metrics --book candide --methods xlm_roberta vader
    """
    parser = argparse.ArgumentParser(
        description="Compute emotional-arc comparison metrics.",
    )
    parser.add_argument(
        "--book",
        type=str,
        metavar="SLUG_OR_TITLE",
        help="Compute metrics for a single book (by slug or title).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Compute metrics for every book in the corpus.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        metavar="METHOD",
        default=None,
        help=(
            "Sentiment methods to evaluate.  "
            f"Defaults to {DEFAULT_METHODS}."
        ),
    )
    args = parser.parse_args(argv)

    if not args.book and not args.run_all:
        parser.error("Provide --book SLUG or --all.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    methods = args.methods

    if args.run_all:
        results = compute_all_book_metrics(methods=methods)
    else:
        book = config.get_book(args.book)
        book_metrics = compute_book_metrics(book, methods=methods)
        results = {book["slug"]: book_metrics} if book_metrics else {}

    if not results:
        logger.warning("No results produced -- check that score files exist "
                        "under %s.", config.PROCESSED_DIR)
        sys.exit(1)

    json_path = save_results_json(results)
    csv_path = save_results_csv(results)

    # Print a quick summary via pandas for terminal readability.
    try:
        df = pd.read_csv(csv_path)
        print("\n" + df.to_string(index=False))
    except Exception:
        logger.debug("Could not render summary table via pandas.", exc_info=True)

    print(f"\nResults saved to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
