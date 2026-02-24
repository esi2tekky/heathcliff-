# Wuthering Heights: Cross-Language Emotional Arc Analysis

## Overview

Comparison of the emotional arcs of two versions of *Wuthering Heights* (Emily Brontë):

1. **English Original** — Project Gutenberg #768 (34 chapters)
2. **French Human Translation** — "Un Amant", translated by T. de Wyzewa, Project Gutenberg #63193 (30 chapters across 2 parts)

The French translator restructured the novel from 34 flat chapters into 2 parts (14 + 12 = 26 body chapters + 4 introductory/framing chapters = 30 total), which accounts for the chapter count mismatch.

---

## Sentiment Methods Used

| Method | Language | Score Range | Notes |
|--------|----------|-------------|-------|
| **VADER** | English only | [-1, 1] | Lexicon-based, compound score. Very polarized on long chapters. |
| **labMT** | French only | [-1, 1] | French word happiness scores from hedonometer.org. Very narrow range. |
| **XLM-RoBERTa** | Both EN & FR | [-1, 1] | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`. The only cross-language comparable method. |

---

## Summary Statistics

| Version | Method | Mean | Std Dev | Min | Max |
|---------|--------|------|---------|-----|-----|
| EN Original | VADER | -0.3680 | 0.8977 | -1.00 | +1.00 |
| EN Original | XLM-RoBERTa | -0.2301 | 0.1262 | -0.60 | +0.13 |
| FR Translation | labMT | +0.0737 | 0.0058 | +0.06 | +0.09 |
| FR Translation | XLM-RoBERTa | -0.5472 | 0.1575 | -0.81 | -0.12 |

### Key Observations

- **VADER is extremely polarized**: scores oscillate between near -1 and near +1 (std = 0.90). This is a known limitation — VADER sums lexicon hits across long documents, causing saturation. Useful for relative chapter ranking within English, but not for nuanced arc shape.
- **labMT has very narrow variance**: all 30 French chapters score between +0.06 and +0.09. The labMT lexicon averages happiness over all recognized words, and with long chapters the mean regresses toward the language-level baseline. The arc is nearly flat — labMT lacks the dynamic range to capture chapter-to-chapter emotional shifts in literary prose.
- **XLM-RoBERTa is the most informative method**: both the EN and FR scores show meaningful variation (std ~0.13–0.16) within a plausible range, making it the primary basis for cross-language comparison.

---

## Cross-Language Comparison (XLM-RoBERTa)

### Systematic Negative Shift in French

The most striking finding: the French translation scores **substantially more negative** than the English original across the entire novel.

- EN Original mean: **-0.23**
- FR Translation mean: **-0.55**
- Average shift: **-0.32 points** (on a [-1, 1] scale)

This is a consistent downward bias, not a chapter-specific effect — the drift plot shows that the French translation is more negative in **every single chapter**.

### Correlation Metrics

| Normalization | Pearson r | p-value | Spearman ρ | p-value | RMSE | Mean Abs Diff |
|---------------|-----------|---------|------------|---------|------|---------------|
| Chapter-truncated (first 30) | 0.336 | 0.070 | 0.391 | 0.033 | 0.360 | 0.324 |
| Percentage-normalized (100 windows) | -0.082 | 0.418 | -0.007 | 0.948 | 0.368 | 0.327 |

### Interpretation

- **Chapter-truncated**: Weak positive correlation (r = 0.34, p = 0.07). The Spearman rank correlation is marginally significant (ρ = 0.39, p = 0.03), suggesting the *relative ordering* of emotional peaks/troughs is partially preserved even though absolute values shift.
- **Percentage-normalized**: Correlation collapses to near zero (r = -0.08). This makes sense — the chapter restructuring (34 → 30) means narrative events don't align at the same percentage position. Resampling destroys whatever structural correspondence existed.
- **RMSE ≈ 0.36**: The average chapter-level discrepancy is about 36% of the full scale, dominated by the systematic negative shift rather than random noise.

---

## Possible Explanations

### 1. Model Calibration Bias (Most Likely)
XLM-RoBERTa was trained on multilingual Twitter data. The model may not be equally calibrated across languages — French text may systematically receive more negative scores regardless of content. This is a **known limitation** of multilingual sentiment models and should be reported as a finding rather than treated as a translation effect.

### 2. Translational Tone Shift
The translator T. de Wyzewa (1862–1917) was known for taking interpretive liberties. A more somber French rendering is plausible, but the *uniformity* of the shift (every chapter is more negative) suggests model bias over translation effect.

### 3. Lexical Properties of French
French literary prose may use vocabulary that the model associates with negative sentiment more readily than equivalent English vocabulary (e.g., formal/literary registers, negation structures).

### 4. Chapter Restructuring
The translator merged/reorganized chapters (34 → 30), which changes what content falls within each "chapter unit" being scored. This explains why percentage normalization destroys correlation.

---

## Plots

All plots saved in `output/plots/`:

| Plot | File | Description |
|------|------|-------------|
| Arc Overlay | `wuthering_heights_xlm_roberta_arc.png` | EN vs FR emotional arcs with smoothing (XLM-RoBERTa) |
| Pct-Normalized | `wuthering_heights_xlm_pct_normalized.png` | Same arcs resampled to 0–100% narrative progress |
| Drift | `wuthering_heights_xlm_drift.png` | Chapter-by-chapter sentiment difference (FR − EN) |
| All Methods | `wuthering_heights_all_methods.png` | 2×2 grid showing all 4 method/version combinations |

---

## Conclusions for Milestone

1. **Translation does shift the inferred emotional arc** — the French human translation of *Wuthering Heights* registers as significantly more negative across all chapters under XLM-RoBERTa.
2. **The arc *shape* is weakly preserved** — Spearman ρ = 0.39 (p = 0.03) on chapter-truncated data suggests peaks and troughs partially align, even though absolute sentiment levels diverge.
3. **Method choice matters enormously**: VADER saturates, labMT flatlines, and only XLM-RoBERTa provides usable cross-language signal. Any paper reporting these results must discuss method limitations.
4. **Chapter restructuring breaks percentage normalization** — when the translator changes chapter boundaries, resampling-based comparison becomes meaningless. Chapter-aligned (truncated) comparison is more honest for this text pair.
5. **Model calibration bias is a confound** — before attributing the negative shift to translation, a control experiment (e.g., scoring parallel sentences of known equivalent sentiment) would be needed to separate model bias from genuine translational tone shift.

---

## Next Steps

- [ ] Add LLM translation (EN → FR via Claude Sonnet) and compare three-way
- [ ] Run LLM Judge sentiment for a model-independent cross-check
- [ ] Repeat analysis on a French-original book (e.g., Candide) to test whether the negative bias reverses direction
- [ ] Investigate XLM-RoBERTa calibration by scoring a set of sentiment-equivalent FR/EN sentence pairs
