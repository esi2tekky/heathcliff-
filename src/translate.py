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
from pathlib import Path

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
# Best-of-N sentiment-guided reranking
# ---------------------------------------------------------------------------

def _translate_chapter_with_temp(
    model: GenerativeModel,
    text: str,
    system_prompt: str,
    temperature: float,
) -> dict:
    """Translate a chapter at a specific temperature."""
    response = model.generate_content(
        f"{system_prompt}\n\n{text}",
        generation_config={
            "temperature": temperature,
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


def _score_text_xlm(text: str) -> float:
    """Score a text with XLM-RoBERTa (imported from sentiment module)."""
    from src.sentiment import _score_xlm_roberta
    return _score_xlm_roberta(text)


def _compute_mini_arc(text: str, n_windows: int = 10) -> list[float]:
    """Compute a mini sentiment arc over a chapter using evenly-spaced windows.

    Splits the chapter into *n_windows* overlapping windows and scores each
    with XLM-RoBERTa, returning a list of scores that represents the
    sentiment trajectory within the chapter.

    Parameters
    ----------
    text : str
        Full chapter text.
    n_windows : int
        Number of evenly-spaced sample points (default 10).

    Returns
    -------
    list[float]
        Sentiment score at each window position.
    """
    from src.sentiment import _xlm_score_single

    words = text.split()
    n_words = len(words)

    # For very short chapters, return a single score.
    if n_words < 50:
        return [_score_text_xlm(text)]

    # Window size: ~20% of the chapter, at least 50 words.
    window_words = max(50, n_words // 5)
    # Ensure we don't have more windows than possible positions.
    n_windows = min(n_windows, max(1, n_words - window_words + 1))

    if n_windows <= 1:
        return [_score_text_xlm(text)]

    scores = []
    for i in range(n_windows):
        start = int(i * (n_words - window_words) / (n_windows - 1))
        end = start + window_words
        chunk = " ".join(words[start:end])
        scores.append(_xlm_score_single(chunk))

    return scores


def _pearson_correlation(a: list[float], b: list[float]) -> float:
    """Compute Pearson correlation between two equal-length lists.

    Returns 0.0 if either series has zero variance (constant values).
    """
    import numpy as np
    a_arr = np.array(a)
    b_arr = np.array(b)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return 0.0
    a_std = np.std(a_arr)
    b_std = np.std(b_arr)
    if a_std == 0 or b_std == 0:
        return 0.0
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def translate_book_best_of_n(
    book: dict,
    n: int = 5,
    temperature: float = 0.8,
    force: bool = False,
) -> dict:
    """Translate with best-of-N shape-guided reranking.

    For each chapter:
      1. Compute a mini sentiment arc of the original text (n_windows sample
         points within the chapter) using XLM-RoBERTa.
      2. Generate N candidate translations in parallel at the given temperature.
      3. Compute a mini arc for each candidate.
      4. Select the candidate whose arc shape (Pearson correlation) most
         closely matches the original's arc — optimizing for trajectory
         fidelity, not magnitude matching.

    Saves to ``{slug}_llm_bon{N}.json`` and also writes to the standard
    ``{slug}_llm.json`` so downstream pipeline phases pick it up.

    Parameters
    ----------
    book : dict
        Book entry from config.BOOKS.
    n : int
        Number of candidate translations per chapter.
    temperature : float
        Sampling temperature for diversity (higher = more diverse).
    force : bool
        If True, retranslate from scratch.

    Returns
    -------
    dict
        Translation result with best-of-N metadata.
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
    bon_path = config.TRANSLATIONS_DIR / f"{slug}_llm_bon{n}.json"
    output_path = config.TRANSLATIONS_DIR / f"{slug}_llm.json"

    # Number of windows per chapter for the mini-arc.
    n_mini_windows = 10

    # ----- Load existing checkpoint or initialise --------------------------
    result = None
    if bon_path.exists() and not force:
        with open(bon_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        logger.info(
            "Resuming best-of-%d %s: %d / %d chapters done.",
            n, slug,
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
            "method": f"best_of_{n}_shape",
            "best_of_n": n,
            "temperature": temperature,
            "selection_criterion": "pearson_correlation",
            "n_chapters": n_chapters,
            "chapters": [None] * n_chapters,
            "metadata": {
                "n_chapters_translated": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            },
        }

    system_prompt = _system_prompt(book)
    vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
    model = GenerativeModel(config.GEMINI_MODEL_TRANSLATE)

    for i, src_chapter in enumerate(source_chapters):
        if result["chapters"][i] is not None:
            logger.debug("Chapter %d already done (best-of-%d), skipping.", i + 1, n)
            continue

        logger.info(
            "Best-of-%d (shape): %s — chapter %d / %d — computing original arc ...",
            n, book["title"], i + 1, n_chapters,
        )

        # 1. Compute the original chapter's mini sentiment arc.
        original_arc = _compute_mini_arc(src_chapter["text"], n_windows=n_mini_windows)
        logger.info(
            "  Original mini-arc (%d points): %s",
            len(original_arc),
            [f"{s:.3f}" for s in original_arc],
        )

        # 2. Generate N candidates in parallel.
        logger.info("  Generating %d candidates at temperature=%.1f ...", n, temperature)
        candidates = []
        total_in = 0
        total_out = 0

        with ThreadPoolExecutor(max_workers=min(n, config.PARALLEL_WORKERS)) as executor:
            futures = [
                executor.submit(
                    _translate_chapter_with_temp,
                    model, src_chapter["text"], system_prompt, temperature,
                )
                for _ in range(n)
            ]
            for future in as_completed(futures):
                try:
                    api_result = future.result()
                    candidates.append(api_result)
                    total_in += api_result["input_tokens"]
                    total_out += api_result["output_tokens"]
                except Exception:
                    logger.exception(
                        "  Candidate generation failed for chapter %d", i + 1,
                    )

        if not candidates:
            logger.error("  No candidates produced for chapter %d; skipping.", i + 1)
            continue

        # 3. Compute mini-arc for each candidate and correlate with original.
        candidate_arcs = []
        candidate_correlations = []
        for j, cand in enumerate(candidates):
            arc = _compute_mini_arc(cand["translated_text"], n_windows=n_mini_windows)
            # Truncate to matching length (should be same, but be safe).
            min_len = min(len(original_arc), len(arc))
            corr = _pearson_correlation(original_arc[:min_len], arc[:min_len])
            candidate_arcs.append(arc)
            candidate_correlations.append(corr)
            logger.debug("  Candidate %d correlation: %.4f", j + 1, corr)

        # 4. Select the candidate with highest shape correlation.
        best_idx = int(max(range(len(candidate_correlations)),
                          key=lambda k: candidate_correlations[k]))
        best_cand = candidates[best_idx]
        best_text = best_cand["translated_text"]

        logger.info(
            "  Candidate correlations: %s",
            [f"{c:.4f}" for c in candidate_correlations],
        )
        logger.info(
            "  Selected candidate %d (correlation=%.4f, best shape match)",
            best_idx + 1, candidate_correlations[best_idx],
        )

        # Also compute scalar scores for metadata/comparison.
        original_scalar = _score_text_xlm(src_chapter["text"])
        selected_scalar = _score_text_xlm(best_text)

        result["chapters"][i] = {
            "number": src_chapter.get("number", i + 1),
            "title": src_chapter.get("title", ""),
            "text": best_text,
            "char_count": len(best_text),
            "word_count": len(best_text.split()),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bon_metadata": {
                "n_candidates": len(candidates),
                "selection_criterion": "pearson_correlation",
                "original_arc": [round(s, 6) for s in original_arc],
                "candidate_correlations": [round(c, 6) for c in candidate_correlations],
                "selected_index": best_idx,
                "selected_correlation": round(candidate_correlations[best_idx], 6),
                "original_score": round(original_scalar, 6),
                "selected_score": round(selected_scalar, 6),
                "selection_diff": round(abs(selected_scalar - original_scalar), 6),
            },
        }

        result["metadata"]["n_chapters_translated"] += 1
        result["metadata"]["total_input_tokens"] += total_in
        result["metadata"]["total_output_tokens"] += total_out

        # Checkpoint to the bon-specific file.
        with open(bon_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(
            "  Chapter %d done — %d candidates, %d total tokens.",
            i + 1, len(candidates), total_in + total_out,
        )

    # Also write to the standard _llm.json so the pipeline picks it up.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        "Best-of-%d (shape) translation complete for %s: %d chapters, "
        "%d total input tokens, %d total output tokens.",
        n, book["title"],
        result["metadata"]["n_chapters_translated"],
        result["metadata"]["total_input_tokens"],
        result["metadata"]["total_output_tokens"],
    )
    return result


# ---------------------------------------------------------------------------
# Temperature sweep translation
# ---------------------------------------------------------------------------

def _estimate_xlm_language_offset(book: dict) -> float:
    """Estimate the systematic XLM-RoBERTa bias between original and human-translated text.

    Scans data/processed/ for matching {slug}_en_original_xlm_roberta.json and
    {slug}_fr_human_xlm_roberta.json pairs. Computes mean per-point difference
    (human_translated - original) across all aligned data.

    For chapter-level files with matching counts, uses those directly.
    Falls back to sliding-window files (always 100 points) for mismatched counts.

    Returns the mean offset (negative for en_to_fr, positive for fr_to_en),
    or 0.0 if no calibration data is found.
    """
    import glob as glob_mod

    direction = book["direction"]
    if direction == "en_to_fr":
        orig_suffix = "en_original"
        human_suffix = "fr_human"
    else:
        orig_suffix = "fr_original"
        human_suffix = "en_human"

    processed = config.PROCESSED_DIR
    all_diffs = []

    # Find all slugs that have both original and human xlm_roberta files.
    orig_pattern = str(processed / f"*_{orig_suffix}_xlm_roberta.json")
    for orig_path in sorted(glob_mod.glob(orig_pattern)):
        # Extract slug from filename.
        fname = Path(orig_path).name
        slug = fname.replace(f"_{orig_suffix}_xlm_roberta.json", "")

        human_path = processed / f"{slug}_{human_suffix}_xlm_roberta.json"
        if not human_path.exists():
            continue

        with open(orig_path, "r", encoding="utf-8") as f:
            orig_data = json.load(f)
        with open(human_path, "r", encoding="utf-8") as f:
            human_data = json.load(f)

        orig_scores = orig_data["scores"]
        human_scores = human_data["scores"]

        if len(orig_scores) == len(human_scores):
            # Chapter counts match — use chapter-level scores.
            diffs = [h - o for h, o in zip(human_scores, orig_scores)]
            all_diffs.extend(diffs)
            logger.info(
                "  Calibration: %s chapter-level (%d points), mean diff=%.4f",
                slug, len(diffs), sum(diffs) / len(diffs),
            )
        else:
            # Mismatched chapter counts — fall back to sliding-window files.
            sw_orig_path = processed / f"{slug}_{orig_suffix}_sw_xlm_roberta.json"
            sw_human_path = processed / f"{slug}_{human_suffix}_sw_xlm_roberta.json"
            if sw_orig_path.exists() and sw_human_path.exists():
                with open(sw_orig_path, "r", encoding="utf-8") as f:
                    sw_orig = json.load(f)
                with open(sw_human_path, "r", encoding="utf-8") as f:
                    sw_human = json.load(f)
                sw_orig_scores = sw_orig["scores"]
                sw_human_scores = sw_human["scores"]
                n = min(len(sw_orig_scores), len(sw_human_scores))
                diffs = [sw_human_scores[j] - sw_orig_scores[j] for j in range(n)]
                all_diffs.extend(diffs)
                logger.info(
                    "  Calibration: %s sliding-window (%d points), mean diff=%.4f",
                    slug, len(diffs), sum(diffs) / len(diffs),
                )
            else:
                logger.debug(
                    "  Calibration: %s chapter mismatch (%d vs %d) and no sw files; skipping.",
                    slug, len(orig_scores), len(human_scores),
                )

    if not all_diffs:
        logger.warning(
            "No calibration data found for direction=%s; using offset=0.0", direction,
        )
        return 0.0

    offset = sum(all_diffs) / len(all_diffs)
    logger.info(
        "Estimated XLM language offset for %s: %.6f (from %d aligned points)",
        direction, offset, len(all_diffs),
    )
    return offset


def translate_book_temp_sweep(
    book: dict,
    force: bool = False,
) -> dict:
    """Translate with temperature sweep: pick the candidate closest to original sentiment.

    For each chapter:
      1. Score the original chapter with XLM-RoBERTa.
      2. Translate in parallel at each temperature in config.TEMP_SWEEP_TEMPS.
      3. Score each candidate translation with XLM-RoBERTa.
      4. Select the candidate minimizing |score_candidate - score_original|.

    Saves to ``{slug}_llm_tempsweep.json`` and also writes to the standard
    ``{slug}_llm.json`` for downstream pipeline compatibility.
    """
    slug = book["slug"]
    orig_lang = config.original_lang(book)
    trans_lang = config.translation_lang(book)
    temps = config.TEMP_SWEEP_TEMPS

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
    sweep_path = config.TRANSLATIONS_DIR / f"{slug}_llm_tempsweep.json"
    output_path = config.TRANSLATIONS_DIR / f"{slug}_llm.json"

    # ----- Load existing checkpoint or initialise --------------------------
    result = None
    if sweep_path.exists() and not force:
        with open(sweep_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        logger.info(
            "Resuming temp-sweep %s: %d / %d chapters done.",
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
            "method": "temp_sweep",
            "temperatures": temps,
            "selection_criterion": "min_abs_sentiment_delta",
            "n_chapters": n_chapters,
            "chapters": [None] * n_chapters,
            "metadata": {
                "n_chapters_translated": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            },
        }

    # Estimate cross-language sentiment bias for calibrated selection.
    offset = _estimate_xlm_language_offset(book)
    logger.info("Calibration offset for temp-sweep: %.6f", offset)

    system_prompt = _system_prompt(book)
    vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
    model = GenerativeModel(config.GEMINI_MODEL_TRANSLATE)

    for i, src_chapter in enumerate(source_chapters):
        if result["chapters"][i] is not None:
            logger.debug("Chapter %d already done (temp-sweep), skipping.", i + 1)
            continue

        logger.info(
            "Temp-sweep: %s — chapter %d / %d — scoring original ...",
            book["title"], i + 1, n_chapters,
        )

        # 1. Score the original chapter and compute adjusted target.
        original_score = _score_text_xlm(src_chapter["text"])
        adjusted_target = original_score + offset
        logger.info("  Original score: %.4f, adjusted target: %.4f (offset=%.4f)",
                     original_score, adjusted_target, offset)

        # 2. Translate in parallel at each temperature.
        logger.info("  Translating at %d temperatures: %s ...", len(temps), temps)
        candidates = [None] * len(temps)
        total_in = 0
        total_out = 0

        with ThreadPoolExecutor(max_workers=min(len(temps), config.PARALLEL_WORKERS)) as executor:
            future_to_idx = {
                executor.submit(
                    _translate_chapter_with_temp,
                    model, src_chapter["text"], system_prompt, temp,
                ): idx
                for idx, temp in enumerate(temps)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    api_result = future.result()
                    candidates[idx] = api_result
                    total_in += api_result["input_tokens"]
                    total_out += api_result["output_tokens"]
                except Exception:
                    logger.exception(
                        "  Translation failed at temp=%.1f for chapter %d",
                        temps[idx], i + 1,
                    )

        # Filter out failed candidates.
        valid = [(idx, c) for idx, c in enumerate(candidates) if c is not None]
        if not valid:
            logger.error("  No candidates produced for chapter %d; skipping.", i + 1)
            continue

        # 3. Score each candidate translation.
        candidate_scores = []
        for idx, cand in valid:
            score = _score_text_xlm(cand["translated_text"])
            candidate_scores.append((idx, score))

        # 4. Select candidate minimizing |score_candidate - adjusted_target|.
        best_idx, best_score = min(
            candidate_scores, key=lambda x: abs(x[1] - adjusted_target)
        )
        best_cand = candidates[best_idx]
        best_text = best_cand["translated_text"]
        best_delta = abs(best_score - adjusted_target)

        # Build score map for metadata.
        all_scores = {f"{temps[idx]:.1f}": round(score, 6) for idx, score in candidate_scores}

        logger.info("  Candidate scores: %s", all_scores)
        logger.info(
            "  Selected temp=%.1f (score=%.4f, delta=%.4f from adjusted target=%.4f)",
            temps[best_idx], best_score, best_delta, adjusted_target,
        )

        result["chapters"][i] = {
            "number": src_chapter.get("number", i + 1),
            "title": src_chapter.get("title", ""),
            "text": best_text,
            "char_count": len(best_text),
            "word_count": len(best_text.split()),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temp_sweep_metadata": {
                "temperatures": temps,
                "original_score": round(original_score, 6),
                "calibration_offset": round(offset, 6),
                "adjusted_target": round(adjusted_target, 6),
                "candidate_scores": all_scores,
                "selected_temperature": temps[best_idx],
                "selected_score": round(best_score, 6),
                "selected_delta": round(best_delta, 6),
            },
        }

        result["metadata"]["n_chapters_translated"] += 1
        result["metadata"]["total_input_tokens"] += total_in
        result["metadata"]["total_output_tokens"] += total_out

        # Checkpoint to the sweep-specific file.
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(
            "  Chapter %d done — %d candidates, %d total tokens.",
            i + 1, len(valid), total_in + total_out,
        )

    # Also write to the standard _llm.json so the pipeline picks it up.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        "Temp-sweep translation complete for %s: %d chapters, "
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
