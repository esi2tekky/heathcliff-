"""Central configuration for the cross-language emotional arc pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRANSLATIONS_DIR = DATA_DIR / "translations"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
BOOKS = [
    {
        "title": "Candide",
        "slug": "candide",
        "author": "Voltaire",
        "fr_ids": [4650],
        "en_id": 19942,
        "direction": "fr_to_en",
        "expected_chapters": 30,
    },
    {
        "title": "Madame Bovary",
        "slug": "madame_bovary",
        "author": "Flaubert",
        "fr_ids": [14155],
        "en_id": 2413,
        "direction": "fr_to_en",
        "expected_chapters": 35,
    },
    {
        "title": "Twenty Thousand Leagues Under the Sea",
        "slug": "twenty_thousand_leagues",
        "author": "Verne",
        "fr_ids": [5097],
        "en_id": 164,
        "direction": "fr_to_en",
        "expected_chapters": 47,
    },
    {
        "title": "Around the World in 80 Days",
        "slug": "around_the_world",
        "author": "Verne",
        "fr_ids": [800],
        "en_id": 103,
        "direction": "fr_to_en",
        "expected_chapters": 37,
    },
    {
        "title": "The Strange Case of Dr. Jekyll and Mr. Hyde",
        "slug": "jekyll_hyde",
        "author": "Stevenson",
        "fr_ids": [76412],
        "en_id": 43,
        "direction": "en_to_fr",
        "expected_chapters": 10,
        "en_chapter_titles": [
            "STORY OF THE DOOR",
            "SEARCH FOR MR. HYDE",
            "DR. JEKYLL WAS QUITE AT EASE",
            "THE CAREW MURDER CASE",
            "INCIDENT OF THE LETTER",
            "INCIDENT OF DR. LANYON",
            "INCIDENT AT THE WINDOW",
            "THE LAST NIGHT",
            "DR. LANYON\u2019S NARRATIVE",
            "HENRY JEKYLL\u2019S FULL STATEMENT OF THE CASE",
        ],
    },
    {
        "title": "Wuthering Heights",
        "slug": "wuthering_heights",
        "author": "Brontë",
        "fr_ids": [63193],
        "en_id": 768,
        "direction": "en_to_fr",
        "expected_chapters": 34,
    },
    {
        "title": "The Picture of Dorian Gray",
        "slug": "dorian_gray",
        "author": "Wilde",
        "fr_ids": [14192],
        "en_id": 174,
        "direction": "en_to_fr",
        "expected_chapters": 21,
    },
    {
        "title": "Treasure Island",
        "slug": "treasure_island",
        "author": "Stevenson",
        "fr_ids": [76225],
        "en_id": 120,
        "direction": "en_to_fr",
        "expected_chapters": 34,
    },
]

# ---------------------------------------------------------------------------
# API configuration (Vertex AI)
# ---------------------------------------------------------------------------
GCP_PROJECT_ID = "focused-studio-487804-s4"
GCP_REGION = "us-central1"

# Gemini
GEMINI_MODEL_TRANSLATE = "gemini-2.5-pro"
GEMINI_MODEL_JUDGE = "gemini-2.0-flash"

TRANSLATE_TEMPERATURE = 0.3
TRANSLATE_MAX_TOKENS = 65536
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
PARALLEL_WORKERS = 20   # max concurrent API calls for parallel translation
TEMP_SWEEP_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0]

# ---------------------------------------------------------------------------
# Sentiment configuration
# ---------------------------------------------------------------------------
XLM_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
XLM_CHUNK_TOKENS = 400
XLM_OVERLAP_TOKENS = 50
LABMT_API_URL = "https://hedonometer.org/api/v1/words/?format=json&wordlist__title=labMT-fr-v2"
LABMT_CACHE_PATH = DATA_DIR / "labmt_fr.json"

# Sliding-window (Reagan et al.) configuration
SW_WINDOW_WORDS = 10_000    # W = 10,000 words per window
SW_N_POINTS = 100           # n = 100 evenly-spaced sample points

# ---------------------------------------------------------------------------
# Smoothing / metrics configuration
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW = 11
SMOOTHING_POLYORDER = 3
PEAK_TOP_K = 3
N_PERCENTAGE_WINDOWS = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_book(title_or_slug: str) -> dict:
    """Look up a book by title or slug (case-insensitive)."""
    key = title_or_slug.lower().strip()
    for book in BOOKS:
        if book["title"].lower() == key or book["slug"].lower() == key:
            return book
    raise ValueError(f"Unknown book: {title_or_slug!r}")


def get_versions(book: dict) -> tuple[str, ...]:
    """Return version labels for all text variants of a book.

    This is the single source of truth for direction handling:
      - Standard (fr_to_en): ("fr_original", "en_human", "en_llm")
      - Reverse  (en_to_fr): ("en_original", "fr_human", "fr_llm")
    """
    if book["direction"] == "en_to_fr":
        return ("en_original", "fr_human", "fr_llm")
    return ("fr_original", "en_human", "en_llm")


def original_lang(book: dict) -> str:
    """Return the language code of the original text."""
    return "en" if book["direction"] == "en_to_fr" else "fr"


def translation_lang(book: dict) -> str:
    """Return the language code of the translated text."""
    return "fr" if book["direction"] == "en_to_fr" else "en"
