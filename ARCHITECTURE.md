# Architecture: Cross-Language Emotional Arc Comparison

## Research Question
Does translating a story into another language change its inferred emotional arc? Can LLMs achieve better emotional fidelity in translation than human translators?

## Corpus

All texts are French originals with English human translations from Project Gutenberg.

### Primary Corpus (use all of these)

| Work | Author | FR Gutenberg ID(s) | EN Gutenberg ID | Notes |
|------|--------|-------------------|-----------------|-------|
| Candide | Voltaire | 4650 | 19942 | Short (~30 chapters), great for testing. Start here. |
| Madame Bovary | Flaubert | 14155 | 2413 | Single volume, clean chapter structure |
| Twenty Thousand Leagues Under the Sea | Verne | 5097 | 164 | Single volume French. Alt French: search "Vingt mille Lieues Sous Les Mers — Complete" |
| Around the World in 80 Days | Verne | 800 (verify) | 103 | Short, episodic, clear chapters |
| Wuthering Heights | Brontë | 63193 (FR translation "Un amant") | 768 (EN original) | **Reverse direction**: EN is original, FR is translation |

### Extended Corpus (add if time permits)

| Work | Author | FR Gutenberg ID(s) | EN Gutenberg ID | Notes |
|------|--------|-------------------|-----------------|-------|
| Les Misérables | Hugo | 17489, 17493, 17494, 17518, 17519 (Tomes I-V) | 135 | Very long. Use Tome I only, or combine all. |
| Count of Monte Cristo | Dumas | 17989, 17990, 17991, 17992 (Tomes I-IV) | 1184 | Also very long, split into tomes |
| Three Musketeers | Dumas | Search "Les trois mousquetaires" | 1257 | Verify EN ID |

### Gutenberg URL Pattern
```
Text: https://www.gutenberg.org/cache/epub/{ID}/pg{ID}.txt
HTML: https://www.gutenberg.org/files/{ID}/{ID}-h/{ID}-h.htm
```

### Important Note on Direction
Most works: French original → English translation (testing: does translating INTO English preserve arc?)
Wuthering Heights: English original → French translation (testing: does translating OUT OF English preserve arc?)
Track this in your metadata. Both directions are valuable.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE PHASES                       │
│                                                         │
│  1. DOWNLOAD ──► 2. CHAPTER SPLIT ──► 3. LLM TRANSLATE │
│                                              │          │
│                                              ▼          │
│  For each work, we now have 3 versions:                 │
│    • French original (chapter-aligned)                  │
│    • English human translation (chapter-aligned)        │
│    • English LLM translation (chapter-aligned)          │
│                                              │          │
│                                              ▼          │
│  4. SENTIMENT ──► 5. METRICS ──► 6. VISUALIZE           │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Download (`src/download.py`)

- Download each text from Gutenberg given an ID
- Cache raw downloads in `data/raw/{id}.txt`
- Strip Gutenberg headers/footers (look for `*** START OF` and `*** END OF` markers)
- Handle encoding (some French texts are ISO-8859-1, some UTF-8)
- Return clean text

```python
# Interface
def download_gutenberg(book_id: int, cache_dir: str = "data/raw") -> str:
    """Download and return cleaned text from Project Gutenberg."""
```

## Phase 2: Chapter Splitting (`src/chapter_split.py`)

This is the trickiest part. Different books use different chapter markers.

### Strategy
1. Try a series of regex patterns in order of specificity
2. Fall back to manual chapter counts if automatic detection fails
3. Store detected chapter boundaries for manual verification

### Chapter Patterns to Detect

**English patterns:**
```
CHAPTER I, CHAPTER 1, CHAPTER ONE
Chapter I, Chapter 1, Chapter One
I., II., III. (at start of line, roman numerals)
BOOK FIRST, BOOK SECOND
Part I, Part II
```

**French patterns:**
```
CHAPITRE I, CHAPITRE PREMIER
Chapitre I, Chapitre 1
I., II., III. (roman numerals)
PREMIÈRE PARTIE, DEUXIÈME PARTIE
LIVRE PREMIER, LIVRE DEUXIÈME
```

### Output Format
```json
{
  "title": "Candide",
  "author": "Voltaire",
  "language": "fr",
  "gutenberg_id": 4650,
  "n_chapters": 30,
  "chapters": [
    {
      "number": 1,
      "title": "Comment Candide fut élevé dans un beau château...",
      "text": "Il y avait en Westphalie, dans le château de M. le baron...",
      "char_count": 2847,
      "word_count": 512
    }
  ]
}
```

### Validation
After splitting, print a comparison table:
```
Candide:  FR chapters: 30  |  EN chapters: 30  ✓
Bovary:   FR chapters: 35  |  EN chapters: 35  ✓
```
If chapter counts don't match, flag for manual review. Small mismatches (±1-2) are OK — some editions merge/split chapters.

---

## Phase 3: LLM Translation (`src/translate.py`)

Translate each French chapter into English using the Anthropic API.

### Config
- Model: `claude-sonnet-4-20250514` (good quality/cost balance for translation)
- Max tokens: 4096 per chapter (most chapters are shorter)
- Temperature: 0.3 (we want faithful translation, not creative)

### System Prompt
```
You are a literary translator specializing in French-to-English translation. 
Translate the following passage of French literature into English.
Preserve the original tone, emotional register, and narrative style as faithfully as possible.
Do not add commentary, notes, or explanations. Output only the English translation.
```

### Implementation Details
- Process chapters sequentially, one API call per chapter
- Save after each chapter (checkpoint/resume if interrupted)
- Save translations to `data/translations/{title}_llm.json` in same format as chapter splits
- Log token usage and cost per work
- Add a 1-second delay between calls to stay within rate limits

### For Wuthering Heights (reverse direction)
Translate the English original into French using the same approach, then compare against the human French translation.

---

## Phase 4: Sentiment Analysis (`src/sentiment.py`)

Run three methods on every chapter of every version.

### Method 1: Lexicon Baseline

**English (VADER):**
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores(chapter_text)['compound']  # [-1, 1]
```

**French (labMT):**
- Download the labMT word happiness scores from: https://hedonometer.org/api/v1/words/?format=json&wordlist__title=labMT-fr-we
- Alternative: Use the FEEL lexicon (French Expanded Emotion Lexicon)
- For each chapter: average the happiness scores of all recognized words, then center so 0 = neutral

```python
def labmt_sentiment(text: str, lexicon: dict) -> float:
    """Score text using labMT word happiness ratings."""
    words = tokenize(text.lower())
    scores = [lexicon[w] for w in words if w in lexicon]
    if not scores:
        return 0.0
    # labMT scores are 1-9 with 5 = neutral. Rescale to [-1, 1]
    return (np.mean(scores) - 5) / 4
```

### Method 2: XLM-RoBERTa (Primary Neural Method)

Use `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` — same model for all languages.

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    top_k=None,  # get all label scores
    truncation=True,
    max_length=512
)
```

For long chapters (>512 tokens):
1. Split chapter into ~400-token chunks with 50-token overlap
2. Score each chunk
3. Average scores across chunks (weighted by chunk length)

Map output labels to a continuous score:
```python
# Model outputs: positive, neutral, negative with confidence scores
# Convert to single value in [-1, 1]:
score = (prob_positive * 1) + (prob_neutral * 0) + (prob_negative * -1)
```

### Method 3: Zero-Shot LLM Scoring (Secondary, Subset Only)

Use `claude-haiku-4-5-20250929` to score emotional valence. Apply to primary corpus only.

```
Rate the overall emotional tone of the following passage on a scale from 
-10 (extremely negative/dark/despairing) to +10 (extremely positive/joyful/uplifting).
Consider the mood, events, character emotions, and narrative atmosphere.
Respond with ONLY a single number, nothing else.
```

- Rescale output from [-10, 10] to [-1, 1]
- This gives us an "LLM as judge" sentiment baseline

### Output Format
For each (work, version, method), store:
```json
{
  "title": "Candide",
  "version": "fr_original",  // or "en_human", "en_llm"
  "method": "xlm_roberta",   // or "vader", "labmt", "llm_judge"
  "scores": [0.23, -0.15, 0.67, ...],  // one per chapter
  "n_chapters": 30
}
```

---

## Phase 5: Metrics (`src/metrics.py`)

### Pairwise Comparisons
For each work, compute metrics between all 3 pairs:
- FR_original ↔ EN_human
- FR_original ↔ EN_LLM  
- EN_human ↔ EN_LLM

### Metrics

```python
def compute_all_metrics(arc_a: np.array, arc_b: np.array) -> dict:
    """Compute all comparison metrics between two emotional arcs."""
    return {
        "pearson_r": pearsonr(arc_a, arc_b)[0],
        "pearson_p": pearsonr(arc_a, arc_b)[1],
        "spearman_rho": spearmanr(arc_a, arc_b)[0],
        "spearman_p": spearmanr(arc_a, arc_b)[1],
        "dtw_distance": compute_dtw(arc_a, arc_b),
        "rmse": np.sqrt(np.mean((arc_a - arc_b) ** 2)),
        "mean_abs_diff": np.mean(np.abs(arc_a - arc_b)),
        "cosine_similarity": cosine_sim(arc_a, arc_b),
        "peak_alignment": compute_peak_alignment(arc_a, arc_b, top_k=3),
        "trough_alignment": compute_trough_alignment(arc_a, arc_b, top_k=3),
    }
```

### Critical Point Alignment
1. Smooth both arcs with Savitzky-Golay (window=11, polyorder=3)
2. Find top-k peaks and troughs in each
3. For each peak in arc_a, find the nearest peak in arc_b
4. Report average positional shift (in % of narrative)

### Secondary: Percentage-Normalized Arcs
As a secondary analysis, also compute 100-window sliding-window arcs (like Reagan et al.) and repeat all metrics on those. This tests whether chapter-level alignment vs percentage-normalization changes conclusions.

---

## Phase 6: Visualization (`src/visualize.py`)

### Per-Work Plots

**Plot 1: Three-Arc Overlay**
- X-axis: Chapter number (or normalized 0-100%)
- Y-axis: Sentiment score [-1, 1]
- Three lines: FR original (red), EN human (blue), EN LLM (green)
- Smoothed with Savitzky-Golay, raw scores as faint background
- Title: "{Work Title} — Emotional Arc Comparison ({Method})"

**Plot 2: Drift Plot**
- X-axis: Chapter number
- Y-axis: Sentiment difference
- Two filled areas: (EN_human - FR_original) in blue, (EN_LLM - FR_original) in green
- Shows WHERE in the narrative translation diverges most

### Summary Plots

**Plot 3: Metrics Heatmap**
- Rows: works
- Columns: metric values for each pair (FR↔EN_human, FR↔EN_LLM)
- Color: similarity (green=high, red=low)
- This is the "headline figure" for the paper

**Plot 4: Scatter — Human vs LLM Translation Fidelity**
- X-axis: correlation(FR, EN_human) per work
- Y-axis: correlation(FR, EN_LLM) per work
- Points labeled by work title
- Diagonal line: above = LLM better, below = human better
- This directly answers: "Does the LLM preserve emotional arc better?"

### Style
- Use matplotlib with a clean style (seaborn-v0_8-whitegrid or similar)
- Font size 12+ for readability
- Save as PNG at 150 DPI
- Also save as PDF for the paper

---

## Pipeline Orchestration (`src/pipeline.py`)

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=[
        "download", "split", "translate", "sentiment", "metrics", "visualize", "all"
    ], default="all")
    parser.add_argument("--works", nargs="+", default=None, 
                        help="Specific works to process (default: all)")
    parser.add_argument("--skip-llm-translate", action="store_true",
                        help="Skip LLM translation phase (use cached if available)")
    parser.add_argument("--sentiment-methods", nargs="+", 
                        default=["vader", "labmt", "xlm_roberta"],
                        help="Which sentiment methods to run")
    args = parser.parse_args()
    
    # ... run phases
```

---

## File Structure
```
project/
├── src/
│   ├── download.py         # Gutenberg download + header stripping
│   ├── chapter_split.py    # Chapter boundary detection  
│   ├── translate.py        # LLM translation via Anthropic API
│   ├── sentiment.py        # VADER, labMT, XLM-RoBERTa, LLM judge
│   ├── metrics.py          # DTW, correlation, critical points
│   ├── visualize.py        # All plotting
│   ├── config.py           # Book metadata, API keys, paths
│   └── pipeline.py         # Orchestrates everything
├── data/
│   ├── raw/                # Downloaded Gutenberg texts
│   ├── processed/          # Cleaned chapter-split JSONs
│   └── translations/       # LLM-generated translations
├── output/
│   ├── plots/              # All generated figures
│   ├── results.json        # Full metrics
│   └── results_table.csv   # Summary table
├── requirements.txt
├── ARCHITECTURE.md         # This file
└── README.md
```

---

## Config (`src/config.py`)

```python
BOOKS = [
    {
        "title": "Candide",
        "author": "Voltaire",
        "fr_ids": [4650],
        "en_id": 19942,
        "direction": "fr_to_en",  # French is original
        "expected_chapters": 30,
    },
    {
        "title": "Madame Bovary", 
        "author": "Flaubert",
        "fr_ids": [14155],
        "en_id": 2413,
        "direction": "fr_to_en",
        "expected_chapters": 35,
    },
    {
        "title": "Twenty Thousand Leagues Under the Sea",
        "author": "Verne",
        "fr_ids": [5097],       # Verify this ID — search for "Vingt mille Lieues" complete
        "en_id": 164,
        "direction": "fr_to_en",
        "expected_chapters": 47,
    },
    {
        "title": "Around the World in 80 Days",
        "author": "Verne",
        "fr_ids": [800],        # Verify — search "Le tour du monde"
        "en_id": 103,
        "direction": "fr_to_en",
        "expected_chapters": 37,
    },
    {
        "title": "Wuthering Heights",
        "author": "Brontë",
        "fr_ids": [63193],      # "Un amant" — FR translation
        "en_id": 768,           # EN original
        "direction": "en_to_fr", # Reverse: English is original
        "expected_chapters": 34,
    },
]

# API config
ANTHROPIC_MODEL_TRANSLATE = "claude-sonnet-4-20250514"
ANTHROPIC_MODEL_JUDGE = "claude-haiku-4-5-20250929"
TRANSLATE_TEMPERATURE = 0.3
TRANSLATE_MAX_TOKENS = 4096

# Sentiment config
XLM_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
SMOOTHING_WINDOW = 11
SMOOTHING_POLYORDER = 3
N_PERCENTAGE_WINDOWS = 100  # For secondary percentage-normalized analysis
```

---

## Dependencies

```
# Core
numpy
scipy
matplotlib
seaborn
pandas

# NLP
nltk
transformers
torch
sentencepiece

# API
anthropic

# Utilities
requests
tqdm
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Chapter counts don't match between FR/EN | Log mismatches. For off-by-one, merge the extra chapter. For larger mismatches, fall back to percentage normalization. |
| LLM translation API costs | Candide is ~30 short chapters ≈ $0.50. Full corpus ≈ $10-20. Budget accordingly. Use Haiku for the judge method. |
| XLM-RoBERTa not calibrated across languages | This is a FINDING, not a bug. Report it. The lexicon baseline cross-validates. |
| Some Gutenberg IDs may be wrong | Verify each ID downloads the right text. Print first 200 chars and check. |
| Long chapters overflow 512-token context | Chunking strategy handles this. Document chunk sizes in results. |
