"""Visualization module for the cross-language emotional arc pipeline.

Generates four plot types:
  1. Three-arc overlay (original vs human vs LLM translation)
  2. Drift plot (sentiment difference from original)
  3. Metrics heatmap (comparison across books)
  4. Fidelity scatter (correlation with original)
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter

from src.config import (
    BOOKS,
    OUTPUT_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    SMOOTHING_POLYORDER,
    SMOOTHING_WINDOW,
    get_book,
    get_versions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
STYLE = "seaborn-v0_8-whitegrid"
FONT_SIZE = 12
DPI = 150

VERSION_COLORS = {
    "original": "#d62728",              # red
    "human_translation": "#1f77b4",     # blue
    "LLM_translation": "#2ca02c",       # green
}

VERSION_LABELS = {
    "original": "Original",
    "human_translation": "Human Translation",
    "LLM_translation": "LLM Translation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    """Apply the common matplotlib style and font settings."""
    plt.style.use(STYLE)
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
    })


def _ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str) -> None:
    """Save a figure as both PNG and PDF inside PLOTS_DIR."""
    _ensure_dirs()
    png_path = PLOTS_DIR / f"{stem}.png"
    pdf_path = PLOTS_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    logger.info("Saved %s and %s", png_path, pdf_path)
    plt.close(fig)


def _load_scores(slug: str, version: str, method: str) -> list[float]:
    """Load sentiment scores from a processed JSON file.

    Expected path: data/processed/{slug}_{version}_{method}.json
    The JSON file should contain a list of per-chapter sentiment scores.
    """
    path = PROCESSED_DIR / f"{slug}_{version}_{method}.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Handle both bare lists and dicts with a "scores" key.
    if isinstance(data, list):
        return [float(v) for v in data]
    if isinstance(data, dict) and "scores" in data:
        return [float(v) for v in data["scores"]]
    raise ValueError(f"Unexpected JSON structure in {path}")


def _smooth(scores: list[float]) -> np.ndarray:
    """Apply Savitzky-Golay smoothing, gracefully handling short arcs."""
    arr = np.asarray(scores, dtype=float)
    window = SMOOTHING_WINDOW
    polyorder = SMOOTHING_POLYORDER
    # Window must be odd and <= length of data; polyorder < window.
    if len(arr) < window:
        window = len(arr) if len(arr) % 2 == 1 else len(arr) - 1
    if window < 1:
        return arr
    if polyorder >= window:
        polyorder = window - 1
    if window < 3 or polyorder < 1:
        return arr
    return savgol_filter(arr, window_length=window, polyorder=polyorder)


# ---------------------------------------------------------------------------
# Plot 1 -- Three-arc overlay
# ---------------------------------------------------------------------------

def plot_arc_overlay(book: dict, method: str) -> None:
    """Plot original, human-translated, and LLM-translated emotional arcs.

    Parameters
    ----------
    book : dict
        A book entry from ``config.BOOKS``.
    method : str
        Sentiment method key (e.g. ``"xlm"``, ``"labmt"``).
    """
    _apply_style()
    slug = book["slug"]
    title = book["title"]
    versions = get_versions(book)  # (original, human, llm)
    role_keys = ("original", "human_translation", "LLM_translation")

    fig, ax = plt.subplots(figsize=(12, 5))

    for version_label, role in zip(versions, role_keys):
        try:
            raw = _load_scores(slug, version_label, method)
        except FileNotFoundError:
            logger.warning("Missing scores for %s / %s / %s -- skipping", slug, version_label, method)
            continue

        x = np.linspace(0, 100, len(raw))
        smoothed = _smooth(raw)
        color = VERSION_COLORS[role]
        label = VERSION_LABELS[role]

        # Faint raw line
        ax.plot(x, raw, color=color, alpha=0.2, linewidth=1)
        # Solid smoothed arc
        ax.plot(x, smoothed, color=color, alpha=1.0, linewidth=2, label=label)

    ax.set_xlabel("Narrative Progress (%)")
    ax.set_ylabel("Sentiment Score")
    # Auto-scale y-axis to data range with 10% padding
    ymin, ymax = ax.get_ylim()
    padding = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.set_title(f"{title} \u2014 Emotional Arc ({method})")
    ax.legend(loc="best")

    _save(fig, f"{slug}_{method}_arc")


# ---------------------------------------------------------------------------
# Plot 2 -- Drift plot
# ---------------------------------------------------------------------------

def plot_drift(book: dict, method: str) -> None:
    """Plot sentiment difference of translations from the original.

    Parameters
    ----------
    book : dict
        A book entry from ``config.BOOKS``.
    method : str
        Sentiment method key.
    """
    _apply_style()
    slug = book["slug"]
    title = book["title"]
    versions = get_versions(book)
    orig_ver = versions[0]
    human_ver = versions[1]
    llm_ver = versions[2]

    try:
        orig_scores = _load_scores(slug, orig_ver, method)
        human_scores = _load_scores(slug, human_ver, method)
        llm_scores = _load_scores(slug, llm_ver, method)
    except FileNotFoundError as exc:
        logger.warning("Missing scores for drift plot (%s / %s): %s", slug, method, exc)
        return

    # Truncate to the shortest arc length.
    min_len = min(len(orig_scores), len(human_scores), len(llm_scores))

    orig = np.asarray(orig_scores[:min_len])
    human = np.asarray(human_scores[:min_len])
    llm = np.asarray(llm_scores[:min_len])

    x = np.linspace(0, 100, min_len)
    human_drift = human - orig
    llm_drift = llm - orig

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        x, 0, human_drift,
        color=VERSION_COLORS["human_translation"], alpha=0.4,
        label="Human Translation \u2212 Original",
    )
    ax.fill_between(
        x, 0, llm_drift,
        color=VERSION_COLORS["LLM_translation"], alpha=0.4,
        label="LLM Translation \u2212 Original",
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Narrative Progress (%)")
    ax.set_ylabel("Sentiment Difference from Original")
    ax.set_title(f"{title} \u2014 Sentiment Drift ({method})")
    ax.legend(loc="best")

    _save(fig, f"{slug}_{method}_drift")


# ---------------------------------------------------------------------------
# Plot 3 -- Metrics heatmap
# ---------------------------------------------------------------------------

# Metrics where lower = better (distances); these get inverted so green = good.
_INVERT_METRICS = {"dtw", "rmse", "mean_abs_diff"}


def _load_results() -> dict:
    """Load the aggregated results JSON."""
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_metrics_heatmap(results: dict, method: str) -> None:
    """Render a heatmap of comparison metrics across books.

    Parameters
    ----------
    results : dict
        The full results dictionary (keyed by slug or structured list).
    method : str
        Sentiment method key.
    """
    _apply_style()

    # --- Build rows and columns from results ---------------------------------
    # Structure from metrics.py:
    #   results[slug][method]["original_vs_human"]["chapter"] -> {metric: value}
    #   results[slug][method]["original_vs_llm"]["chapter"]   -> {metric: value}

    pairs = ("original_vs_human", "original_vs_llm")
    pair_labels = {"original_vs_human": "Orig\u2194Human", "original_vs_llm": "Orig\u2194LLM"}

    # Discover metric names from the first available entry.
    metric_names: list[str] = []
    for slug_data in results.values():
        if not isinstance(slug_data, dict):
            continue
        method_data = slug_data.get(method, {})
        for pair in pairs:
            pair_data = method_data.get(pair, {})
            # Navigate into "chapter" sub-dict if present
            chapter_data = pair_data.get("chapter", pair_data)
            if chapter_data and isinstance(chapter_data, dict):
                metric_names = sorted(chapter_data.keys())
                break
        if metric_names:
            break

    if not metric_names:
        logger.warning("No metrics found for method %s -- skipping heatmap", method)
        return

    # Column labels: "metric (pair)"
    columns: list[str] = []
    for metric in metric_names:
        for pair in pairs:
            columns.append(f"{metric}\n({pair_labels[pair]})")

    row_labels: list[str] = []
    matrix: list[list[float]] = []

    for book in BOOKS:
        slug = book["slug"]
        slug_data = results.get(slug, {})
        method_data = slug_data.get(method, {})
        if not method_data:
            continue
        row: list[float] = []
        for metric in metric_names:
            for pair in pairs:
                pair_data = method_data.get(pair, {})
                chapter_data = pair_data.get("chapter", pair_data)
                val = chapter_data.get(metric, float("nan")) if isinstance(chapter_data, dict) else float("nan")
                # Invert distance metrics so that green = good across the board.
                if metric.lower() in _INVERT_METRICS:
                    val = -val
                row.append(val)
        row_labels.append(book["title"])
        matrix.append(row)

    if not matrix:
        logger.warning("Empty matrix for heatmap (%s) -- skipping", method)
        return

    data = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 1.2), max(4, len(row_labels) * 0.8)))
    sns.heatmap(
        data,
        xticklabels=columns,
        yticklabels=row_labels,
        cmap="RdYlGn",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Comparison Metrics \u2014 {method}")
    fig.tight_layout()

    _save(fig, f"{method}_heatmap")


# ---------------------------------------------------------------------------
# Plot 4 -- Fidelity scatter
# ---------------------------------------------------------------------------

def plot_fidelity_scatter(results: dict, method: str) -> None:
    """Scatter of correlation(original, human) vs correlation(original, LLM).

    Parameters
    ----------
    results : dict
        The full results dictionary.
    method : str
        Sentiment method key.
    """
    _apply_style()

    titles: list[str] = []
    human_corrs: list[float] = []
    llm_corrs: list[float] = []

    for book in BOOKS:
        slug = book["slug"]
        slug_data = results.get(slug, {})
        method_data = slug_data.get(method, {})
        # Navigate: original_vs_human -> chapter -> pearson_r
        h_pair = method_data.get("original_vs_human", {})
        l_pair = method_data.get("original_vs_llm", {})
        h_chapter = h_pair.get("chapter", h_pair)
        l_chapter = l_pair.get("chapter", l_pair)
        h_corr = h_chapter.get("pearson_r") if isinstance(h_chapter, dict) else None
        l_corr = l_chapter.get("pearson_r") if isinstance(l_chapter, dict) else None
        if h_corr is None or l_corr is None:
            logger.warning("Missing correlation for %s / %s -- skipping point", slug, method)
            continue
        titles.append(book["title"])
        human_corrs.append(float(h_corr))
        llm_corrs.append(float(l_corr))

    if not titles:
        logger.warning("No data for fidelity scatter (%s) -- skipping", method)
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(human_corrs, llm_corrs, s=80, zorder=3, color="#333333")

    # Label each point
    for label, x, y in zip(titles, human_corrs, llm_corrs):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=FONT_SIZE - 2,
        )

    # Diagonal reference line (y = x)
    lo = min(min(human_corrs), min(llm_corrs)) - 0.05
    hi = max(max(human_corrs), max(llm_corrs)) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")

    ax.set_xlabel("Correlation (Original, Human Translation)")
    ax.set_ylabel("Correlation (Original, LLM Translation)")
    ax.set_title(f"Fidelity Scatter \u2014 {method}")
    ax.legend(loc="lower right")

    _save(fig, f"{method}_fidelity")


# ---------------------------------------------------------------------------
# Plot 5 -- Best-of-N comparison
# ---------------------------------------------------------------------------

BON_COLORS = {
    "N=1": "#2ca02c",       # green (baseline LLM)
    "N=5": "#ff7f0e",       # orange
    "N=10": "#9467bd",      # purple
}


def plot_bon_comparison(book: dict, method: str, n_values: list[int] | None = None) -> None:
    """Compare emotional arcs across different best-of-N settings.

    Overlays the original arc with LLM translations at N=1, N=5, N=10 etc.
    to visualize how reranking improves emotional fidelity.

    Parameters
    ----------
    book : dict
        A book entry from ``config.BOOKS``.
    method : str
        Sentiment method key (e.g. ``"sw_xlm_roberta"``).
    n_values : list[int] or None
        N values to compare.  Defaults to [1, 5, 10].
    """
    from src.config import TRANSLATIONS_DIR

    if n_values is None:
        n_values = [1, 5, 10]

    _apply_style()
    slug = book["slug"]
    title = book["title"]
    versions = get_versions(book)
    orig_ver = versions[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    ax_arc, ax_diff = axes

    # --- Original arc ---
    try:
        orig_raw = _load_scores(slug, orig_ver, method)
    except FileNotFoundError:
        logger.warning("Missing original scores for %s / %s", slug, method)
        plt.close(fig)
        return

    x_orig = np.linspace(0, 100, len(orig_raw))
    orig_smooth = _smooth(orig_raw)
    ax_arc.plot(x_orig, orig_smooth, color="#d62728", linewidth=2.5, label="Original", zorder=10)

    # --- Human translation ---
    human_ver = versions[1]
    try:
        human_raw = _load_scores(slug, human_ver, method)
        x_h = np.linspace(0, 100, len(human_raw))
        human_smooth = _smooth(human_raw)
        ax_arc.plot(x_h, human_smooth, color="#1f77b4", linewidth=2, alpha=0.7,
                    label="Human Translation", linestyle="--")
    except FileNotFoundError:
        pass

    # --- LLM arcs at different N values ---
    llm_ver = versions[2]  # e.g. "fr_llm" or "en_llm"
    arcs_for_diff: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for n_val in n_values:
        # Check if we have a bon-specific translation file
        if n_val == 1:
            # N=1 is the baseline — check for _llm_n1.json backup first,
            # then fall back to the standard processed scores
            bon_path = TRANSLATIONS_DIR / f"{slug}_llm_n1.json"
            if bon_path.exists():
                # Need to load scores from processed dir tagged with bon1
                score_label = llm_ver
            else:
                score_label = llm_ver
        else:
            # For N>1, check if bon-specific scores exist
            # The bon translation overwrites _llm.json, so scores are
            # tagged with the standard llm version label
            bon_path = TRANSLATIONS_DIR / f"{slug}_llm_bon{n_val}.json"
            if not bon_path.exists():
                logger.info("No best-of-%d translation found for %s", n_val, slug)
                continue
            score_label = llm_ver

        try:
            raw = _load_scores(slug, score_label, method)
        except FileNotFoundError:
            logger.info("No scores for %s / %s / %s (N=%d)", slug, score_label, method, n_val)
            continue

        label = f"N={n_val}"
        color = BON_COLORS.get(label, f"C{n_val}")
        x = np.linspace(0, 100, len(raw))
        smoothed = _smooth(raw)

        ax_arc.plot(x, smoothed, color=color, linewidth=2, label=f"LLM ({label})")

        # Store for difference subplot
        min_len = min(len(orig_smooth), len(smoothed))
        arcs_for_diff[label] = (
            np.interp(np.linspace(0, 100, min_len), x_orig, orig_smooth),
            np.interp(np.linspace(0, 100, min_len), x, smoothed),
        )

    ax_arc.set_xlabel("Narrative Progress (%)")
    ax_arc.set_ylabel("Sentiment Score")
    ax_arc.set_title(f"{title} — Best-of-N Comparison ({method})")
    ax_arc.legend(loc="best")

    # Auto-scale y-axis
    ymin, ymax = ax_arc.get_ylim()
    padding = (ymax - ymin) * 0.1
    ax_arc.set_ylim(ymin - padding, ymax + padding)

    # --- Difference subplot: |LLM - Original| for each N ---
    for label, (orig_interp, llm_interp) in arcs_for_diff.items():
        x_diff = np.linspace(0, 100, len(orig_interp))
        abs_diff = np.abs(llm_interp - orig_interp)
        color = BON_COLORS.get(label, "grey")
        ax_diff.fill_between(x_diff, 0, abs_diff, color=color, alpha=0.3, label=label)
        ax_diff.plot(x_diff, abs_diff, color=color, linewidth=1.5)

    ax_diff.set_xlabel("Narrative Progress (%)")
    ax_diff.set_ylabel("|LLM − Original|")
    ax_diff.set_title("Absolute Sentiment Deviation from Original")
    ax_diff.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_diff.legend(loc="best")

    fig.tight_layout()
    _save(fig, f"{slug}_{method}_bon_comparison")


def plot_bon_summary_table(book: dict, method: str, n_values: list[int] | None = None) -> None:
    """Generate a summary bar chart of per-chapter sentiment fidelity by N.

    For each N value, computes mean |LLM_score - Original_score| across chapters
    using the bon_metadata stored in the translation files.

    Parameters
    ----------
    book : dict
        A book entry from ``config.BOOKS``.
    method : str
        Sentiment method key.
    n_values : list[int] or None
        N values to compare. Defaults to [1, 5, 10].
    """
    from src.config import TRANSLATIONS_DIR

    if n_values is None:
        n_values = [1, 5, 10]

    _apply_style()
    slug = book["slug"]
    title = book["title"]

    mean_corrs: dict[str, float] = {}
    mean_diffs: dict[str, float] = {}

    for n_val in n_values:
        if n_val == 1:
            bon_path = TRANSLATIONS_DIR / f"{slug}_llm_n1.json"
            if not bon_path.exists():
                bon_path = TRANSLATIONS_DIR / f"{slug}_llm_bon1.json"
        else:
            bon_path = TRANSLATIONS_DIR / f"{slug}_llm_bon{n_val}.json"

        if not bon_path.exists():
            continue

        with open(bon_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        corrs = []
        diffs = []
        for ch in data.get("chapters", []):
            if ch is None:
                continue
            meta = ch.get("bon_metadata", {})
            if meta:
                if "selected_correlation" in meta:
                    corrs.append(meta["selected_correlation"])
                if "selection_diff" in meta:
                    diffs.append(meta["selection_diff"])

        label = f"N={n_val}"
        if corrs:
            mean_corrs[label] = float(np.mean(corrs))
        if diffs:
            mean_diffs[label] = float(np.mean(diffs))

    if not mean_corrs and not mean_diffs:
        logger.info("No best-of-N data available for %s", slug)
        return

    # Use correlation if available, otherwise fall back to magnitude diff.
    if mean_corrs:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(mean_corrs.keys())
        values = [mean_corrs[k] for k in labels]
        colors = [BON_COLORS.get(k, "grey") for k in labels]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=FONT_SIZE - 1)

        ax.set_xlabel("Best-of-N Setting")
        ax.set_ylabel("Mean Pearson Correlation with Original Arc")
        ax.set_title(f"{title} \u2014 Shape Fidelity by N ({method})")
        ax.set_ylim(0, 1.0)
        fig.tight_layout()
        _save(fig, f"{slug}_{method}_bon_fidelity")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(mean_diffs.keys())
        values = [mean_diffs[k] for k in labels]
        colors = [BON_COLORS.get(k, "grey") for k in labels]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=FONT_SIZE - 1)

        ax.set_xlabel("Best-of-N Setting")
        ax.set_ylabel("Mean |Selected Score \u2212 Original Score|")
        ax.set_title(f"{title} \u2014 Sentiment Fidelity Gap by N ({method})")
        ax.set_ylim(0, max(values) * 1.3 if values else 1)
        fig.tight_layout()
        _save(fig, f"{slug}_{method}_bon_fidelity")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def visualize_all(methods: list[str] | None = None) -> None:
    """Generate every plot for every book and method.

    Parameters
    ----------
    methods : list[str] or None
        Sentiment method keys to visualize.  If ``None``, auto-detect from
        available processed files.
    """
    if methods is None:
        methods = _detect_methods()
    if not methods:
        logger.error("No methods specified or detected -- nothing to plot.")
        return

    logger.info("Generating plots for methods: %s", methods)

    # Per-book plots
    for book in BOOKS:
        for method in methods:
            logger.info("Plotting arcs and drift for %s / %s", book["slug"], method)
            try:
                plot_arc_overlay(book, method)
            except Exception:
                logger.exception("Failed arc overlay for %s / %s", book["slug"], method)
            try:
                plot_drift(book, method)
            except Exception:
                logger.exception("Failed drift plot for %s / %s", book["slug"], method)
            # Best-of-N comparison (only generates if bon data exists)
            try:
                plot_bon_comparison(book, method)
            except Exception:
                logger.exception("Failed bon comparison for %s / %s", book["slug"], method)
            try:
                plot_bon_summary_table(book, method)
            except Exception:
                logger.exception("Failed bon fidelity for %s / %s", book["slug"], method)

    # Cross-book plots (require results.json)
    try:
        results = _load_results()
    except FileNotFoundError:
        logger.warning("output/results.json not found -- skipping heatmap and scatter plots")
        return

    for method in methods:
        logger.info("Plotting heatmap and scatter for %s", method)
        try:
            plot_metrics_heatmap(results, method)
        except Exception:
            logger.exception("Failed heatmap for %s", method)
        try:
            plot_fidelity_scatter(results, method)
        except Exception:
            logger.exception("Failed fidelity scatter for %s", method)


def _detect_methods() -> list[str]:
    """Detect available sentiment methods from processed files.

    Reads the ``"method"`` key from each JSON file to support multi-part
    method names like ``sw_xlm_roberta``.
    """
    methods: set[str] = set()
    if not PROCESSED_DIR.exists():
        return []
    for path in PROCESSED_DIR.glob("*.json"):
        # Skip chapter-split files (they have a "chapters" key, not "method").
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "method" in data:
                methods.add(data["method"])
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(methods)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for visualization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate emotional-arc visualizations.",
    )
    parser.add_argument(
        "--book",
        type=str,
        default=None,
        help="Slug or title of a single book to plot (default: all books).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Generate all plots for all books and methods.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Sentiment method(s) to visualize (e.g. xlm labmt). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    methods = args.methods

    if args.run_all or args.book is None:
        visualize_all(methods=methods)
        return

    # Single-book mode
    book = get_book(args.book)
    if methods is None:
        methods = _detect_methods()
    if not methods:
        logger.error("No methods specified or detected.")
        return

    # Try to load results for cross-book plots even in single-book mode
    results: dict | None = None
    try:
        results = _load_results()
    except FileNotFoundError:
        pass

    for method in methods:
        try:
            plot_arc_overlay(book, method)
        except Exception:
            logger.exception("Failed arc overlay for %s / %s", book["slug"], method)
        try:
            plot_drift(book, method)
        except Exception:
            logger.exception("Failed drift plot for %s / %s", book["slug"], method)

        if results is not None:
            try:
                plot_metrics_heatmap(results, method)
            except Exception:
                logger.exception("Failed heatmap for %s", method)
            try:
                plot_fidelity_scatter(results, method)
            except Exception:
                logger.exception("Failed fidelity scatter for %s", method)


if __name__ == "__main__":
    main()
