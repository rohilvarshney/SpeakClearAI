from html import escape
from pathlib import Path
from datetime import datetime
import random
import tempfile

import joblib
import librosa
import numpy as np
import pandas as pd
import requests
import streamlit as st

MODEL_PATH = Path("models/speakclear_random_forest.joblib")
RESULTS_DIR = Path("results")
PROGRESS_LOG = Path("practice_progress_log.csv")
WORD_BANK_PATH = Path("data/word_bank_expanded.csv")
ALPHABET_WORD_BANK_PATH = Path("data/alphabet_word_bank.csv")
OLLAMA_URL = "http://localhost:11434/api/generate"
SOUND_PRACTICE_SOUNDS = ["s", "z", "r", "th"]
ALPHABET_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

PAGES = [
    "Home",
    "Practice Studio",
    "Try the Demo",
    "Progress Tracker",
    "Research Results",
    "About & Limitations",
]

st.set_page_config(
    page_title="SpeakClear AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Built-in fallback data
# -----------------------------

R_WORDS = [
    "red", "race", "right", "river", "around",
    "carrot", "mirror", "story", "car", "far",
]
TH_WORDS = [
    "think", "thin", "three", "thirty", "thank",
    "birthday", "nothing", "healthy", "bath", "teeth",
]
S_WORDS = [
    "sun", "seal", "seven", "sister", "science",
    "soup", "same", "bus", "yes", "class",
]
Z_WORDS = [
    "zoo", "zero", "zebra", "zipper", "zone",
    "buzz", "lazy", "puzzle", "music", "cheese",
]
SENTENCES = {
    "s": [
        "sally sells seashells",
        "seven sisters sat silently",
        "science sometimes sounds stressful",
        "the sun set slowly",
        "six students studied science",
    ],
    "z": [
        "zebra zoomed through the zoo",
        "zero zebras were in the zone",
        "lazy lizards love music",
        "please pass the cheese",
        "the puzzle was amazing",
    ],
    "mixed": [
        "susan saw the zebra at the zoo",
        "six roses were beside the cheese",
        "the music sounded soft and fuzzy",
        "seven students solved the puzzle",
        "zach said science was easy",
    ],
}
WORD_BANK = {
    "s": S_WORDS,
    "z": Z_WORDS,
    "r": R_WORDS,
    "th": TH_WORDS,
    "mixed": SENTENCES["mixed"],
}
SENTENCE_TARGETS = {
    "sally sells seashells": "s",
    "seven sisters sat silently": "s",
    "science sometimes sounds stressful": "s",
    "the sun set slowly": "s",
    "six students studied science": "s",
    "zebra zoomed through the zoo": "z",
    "zero zebras were in the zone": "z",
    "lazy lizards love music": "z",
    "please pass the cheese": "z",
    "the puzzle was amazing": "z",
    "susan saw the zebra at the zoo": "mixed",
    "six roses were beside the cheese": "mixed",
    "the music sounded soft and fuzzy": "mixed",
    "seven students solved the puzzle": "mixed",
    "zach said science was easy": "mixed",
}
SOUND_DESCRIPTIONS = {
    "s": "Practice a steady /s/ sound in words like sun, seal, and science.",
    "z": "Practice a voiced /z/ sound in words like zero, zoo, and buzz.",
    "r": "Practice an /r/ sound in words like right, river, mirror, and story.",
    "th": "Practice a /th/ sound in words like think, thank, bath, and teeth.",
    "mixed": "Practice short sentences that mix /s/ and /z/ sounds.",
}
SOUND_LABELS = {
    "s": "/s/",
    "z": "/z/",
    "r": "/r/",
    "th": "/th/",
    "mixed": "mixed /s/ and /z/",
}
ALPHABET_FALLBACK = {
    "A": ["apple", "ant", "ask", "able", "after"],
    "B": ["ball", "book", "blue", "baby", "bird"],
    "C": ["cat", "cup", "car", "cold", "city"],
    "D": ["dog", "day", "door", "desk", "dark"],
    "E": ["egg", "eat", "each", "easy", "end"],
    "F": ["fish", "fun", "fast", "four", "face"],
    "G": ["go", "game", "green", "good", "gift"],
    "H": ["hat", "home", "hand", "happy", "help"],
    "I": ["ice", "in", "idea", "into", "island"],
    "J": ["jump", "jam", "job", "just", "joke"],
    "K": ["key", "kite", "kind", "keep", "king"],
    "L": ["lamp", "look", "love", "long", "leaf"],
    "M": ["man", "moon", "milk", "make", "map"],
    "N": ["name", "new", "nice", "near", "night"],
    "O": ["open", "old", "over", "only", "orange"],
    "P": ["pen", "play", "park", "pink", "paper"],
    "Q": ["queen", "quick", "quiet", "quiz", "quest"],
    "R": ["red", "run", "rain", "read", "room"],
    "S": ["sun", "sit", "same", "soft", "school"],
    "T": ["top", "time", "talk", "tree", "table"],
    "U": ["up", "under", "use", "unit", "uncle"],
    "V": ["van", "very", "voice", "visit", "vase"],
    "W": ["water", "walk", "wind", "word", "window"],
    "X": ["xray", "xenon", "xerox", "xylem"],
    "Y": ["yes", "you", "year", "yellow", "young"],
    "Z": ["zoo", "zero", "zip", "zone", "zebra"],
}

# -----------------------------
# Session defaults
# -----------------------------

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "large_text" not in st.session_state:
    st.session_state.large_text = False
if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False
if "progress_df" not in st.session_state:
    st.session_state.progress_df = None

# Sync the sidebar widget before it is created. This avoids StreamlitAPIException
# from writing to a widget key after that widget already exists.
if st.session_state.pop("_nav_pending", False):
    st.session_state.nav_choice = st.session_state.current_page
elif "nav_choice" not in st.session_state:
    st.session_state.nav_choice = st.session_state.current_page


def go_to(page_name):
    st.session_state.current_page = page_name
    st.session_state._nav_pending = True
    st.rerun()


# -----------------------------
# Accessibility and styling
# -----------------------------

def apply_styles():
    large_text = bool(st.session_state.large_text)
    high_contrast = bool(st.session_state.high_contrast)

    base_font = "21px" if large_text else "17px"
    small_font = "18px" if large_text else "15px"
    line_height = "1.65" if large_text else "1.55"

    if high_contrast:
        bg = "#000000"
        panel = "#111111"
        text = "#ffffff"
        muted = "#f3f3f3"
        border = "#ffffff"
        accent = "#ffdd00"
        soft = "#1a1a1a"
        success_bg = "#003b18"
        warning_bg = "#3d3200"
        error_bg = "#3b0000"
        shadow = "none"
        focus = "#ffdd00"
    else:
        bg = "#f4f7fb"
        panel = "#ffffff"
        text = "#111827"
        muted = "#4b5563"
        border = "#d7dee8"
        accent = "#1d4ed8"
        soft = "#eef3ff"
        success_bg = "#e8f7ef"
        warning_bg = "#fff7dc"
        error_bg = "#ffe8e8"
        shadow = "0 10px 28px rgba(17, 24, 39, 0.06)"
        focus = "#1d4ed8"

    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-size: {base_font};
            line-height: {line_height};
        }}

        .stApp {{
            background: {bg};
            color: {text};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {text} !important;
            letter-spacing: -0.02em;
            line-height: 1.25;
        }}

        p, li, label, span, div {{
            color: {text};
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }}

        section[data-testid="stSidebar"] {{
            background: {panel};
            border-right: 1px solid {border};
        }}

        .skip-link {{
            position: absolute;
            left: -999px;
            top: 12px;
            z-index: 1000;
            background: {accent};
            color: {"#000000" if high_contrast else "#ffffff"};
            padding: 10px 14px;
            border-radius: 10px;
            font-weight: 700;
        }}

        .skip-link:focus {{
            left: 16px;
        }}

        a {{
            color: {accent};
            text-decoration: underline;
        }}

        *:focus-visible {{
            outline: 3px solid {focus} !important;
            outline-offset: 3px !important;
        }}

        .app-hero, .compact-header, .section-card, .mini-card, .sound-card, .result-card {{
            background: {panel};
            border: 1px solid {border};
            box-shadow: {shadow};
            color: {text};
        }}

        .app-hero {{
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 18px;
        }}

        .compact-header {{
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }}

        .hero-kicker {{
            font-size: {small_font};
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {accent};
            margin-bottom: 8px;
        }}

        .hero-title {{
            font-size: {"2.5rem" if large_text else "2.2rem"};
            font-weight: 800;
            margin: 0 0 8px 0;
            color: {text};
        }}

        .hero-subtitle {{
            font-size: {"1.2rem" if large_text else "1.05rem"};
            line-height: {line_height};
            color: {muted};
            max-width: 860px;
            margin: 0;
        }}

        .section-card {{
            border-radius: 20px;
            padding: 22px;
            margin-bottom: 16px;
            min-height: 150px;
        }}

        .mini-card, .sound-card, .result-card {{
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 14px;
        }}

        .sound-card {{
            border: 2px solid {border};
            min-height: 170px;
        }}

        .sound-card strong {{
            font-size: 1.35rem;
            color: {accent};
        }}

        .badge {{
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: {small_font};
            font-weight: 700;
            border: 1px solid {border};
            background: {soft};
            color: {text};
            margin: 0 8px 8px 0;
        }}

        .safe-box, .success-box, .error-box, .info-box {{
            border-radius: 16px;
            padding: 16px 18px;
            margin: 16px 0;
            border-left: 8px solid {accent};
        }}

        .safe-box {{
            background: {warning_bg};
        }}

        .success-box {{
            background: {success_bg};
            border-left-color: #16a34a;
        }}

        .error-box {{
            background: {error_bg};
            border-left-color: #dc2626;
        }}

        .info-box {{
            background: {soft};
        }}

        .muted {{
            color: {muted};
            font-size: {small_font};
        }}

        .footer-note {{
            color: {muted};
            font-size: {small_font};
            margin-top: 28px;
            border-top: 1px solid {border};
            padding-top: 16px;
        }}

        button[kind="primary"], .stButton button, .stDownloadButton button {{
            min-height: 44px;
            border-radius: 12px !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stMetric"] {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 16px;
            box-shadow: {shadow};
            overflow: hidden;
        }}

        div[data-testid="stMetricValue"] {{
            font-size: 1.25rem !important;
            line-height: 1.3 !important;
            white-space: normal;
            overflow-wrap: anywhere;
        }}

        div[data-testid="stMetricLabel"] {{
            white-space: normal !important;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 16px;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            flex-wrap: wrap;
        }}

        .stTabs [data-baseweb="tab"] {{
            min-height: 44px;
            border-radius: 12px;
            padding: 10px 16px;
            border: 1px solid {border};
            background: {panel};
        }}

        .stTabs [aria-selected="true"] {{
            background: {soft};
            border: 2px solid {accent};
        }}

        div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label {{
            min-height: 40px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_notice():
    st.markdown(
        """
        <div class="safe-box" role="note">
            <strong>Safety note:</strong> SpeakClear AI is for non-clinical speech practice only.
            It is not a diagnosis, treatment tool, or replacement for a speech-language pathologist.
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(title, body):
    st.markdown(
        f"""
        <div class="info-box">
            <strong>{escape(title)}</strong>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_banner(prediction):
    pred = str(prediction).lower()
    if pred == "clear":
        st.markdown(
            """
            <div class="success-box" role="status">
                <strong>Prediction: Clear</strong>
                <p>The model thinks this recording sounds clear. This is practice feedback only.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="error-box" role="status">
                <strong>Prediction: Unclear</strong>
                <p>The model thinks this recording may need more practice. This is practice feedback only.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# Data helpers
# -----------------------------

def load_csv_if_exists(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _clean_word_bank(df, extra_normalizers=None):
    df = df.copy()
    df["word"] = df["word"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["difficulty"] = df["difficulty"].astype(str).str.strip().str.lower()
    df["notes"] = df["notes"].fillna("").astype(str)
    if extra_normalizers:
        for column, normalizer in extra_normalizers.items():
            if column in df.columns:
                df[column] = df[column].map(normalizer)
    df = df[df["word"].ne("") & df["word"].str.lower().ne("nan")]
    return df.reset_index(drop=True)


@st.cache_data
def load_sound_word_bank():
    df = load_csv_if_exists(WORD_BANK_PATH)
    if df is None:
        return None
    needed = {"word", "target_sound", "source", "difficulty", "notes"}
    if not needed.issubset(set(df.columns)):
        return None
    return _clean_word_bank(
        df,
        extra_normalizers={
            "target_sound": lambda value: str(value).strip().lower(),
        },
    )


@st.cache_data
def load_alphabet_word_bank():
    df = load_csv_if_exists(ALPHABET_WORD_BANK_PATH)
    if df is None:
        return None
    needed = {"word", "letter", "source", "difficulty", "notes"}
    if not needed.issubset(set(df.columns)):
        return None
    return _clean_word_bank(
        df,
        extra_normalizers={
            "letter": lambda value: str(value).strip().upper(),
        },
    )


def fallback_sound_word_bank_df():
    rows = []
    for sound, words in WORD_BANK.items():
        if sound not in SOUND_PRACTICE_SOUNDS and sound != "mixed":
            continue
        for word in words:
            rows.append(
                {
                    "word": word,
                    "target_sound": sound,
                    "source": "built_in",
                    "difficulty": "medium",
                    "notes": SOUND_DESCRIPTIONS.get(sound, ""),
                }
            )
    return pd.DataFrame(rows)


def fallback_alphabet_word_bank_df():
    rows = []
    for letter, words in ALPHABET_FALLBACK.items():
        for word in words:
            rows.append(
                {
                    "word": word,
                    "letter": letter,
                    "source": "built_in",
                    "difficulty": "easy" if len(word) <= 5 else "medium",
                    "notes": f"Practice a word that starts with the letter {letter}.",
                }
            )
    return pd.DataFrame(rows)


def get_sound_word_bank():
    df = load_sound_word_bank()
    if df is None or df.empty:
        return fallback_sound_word_bank_df(), False
    return df, True


def get_alphabet_word_bank():
    df = load_alphabet_word_bank()
    if df is None or df.empty:
        return fallback_alphabet_word_bank_df(), False
    return df, True


def get_word_bank_df():
    return get_sound_word_bank()


def words_for_sound(sound, word_bank_df):
    if word_bank_df is not None and not word_bank_df.empty:
        matched = word_bank_df[word_bank_df["target_sound"] == sound]["word"].tolist()
        if matched:
            return matched
    return WORD_BANK.get(sound, [])


# -----------------------------
# Core ML and feedback
# -----------------------------

def normalize_uploaded_name(filename):
    stem = Path(str(filename)).stem.lower()
    if "-" in stem:
        item = stem.rsplit("-", 1)[0]
    else:
        item = stem
    return item.replace("_", " ").strip()


def infer_target_sound(filename, word_bank_df=None):
    item = normalize_uploaded_name(filename)
    first_word = item.split()[0] if item.split() else item

    if item in SENTENCE_TARGETS:
        return item, SENTENCE_TARGETS[item]

    if word_bank_df is not None and not word_bank_df.empty:
        words = word_bank_df["word"].astype(str).str.lower()
        exact = word_bank_df[words == item]
        if not exact.empty:
            return item, str(exact.iloc[0]["target_sound"]).lower()
        first_match = word_bank_df[words == first_word]
        if not first_match.empty and " " not in str(first_match.iloc[0]["word"]):
            return item, str(first_match.iloc[0]["target_sound"]).lower()

    if item in R_WORDS or first_word in R_WORDS:
        return item, "r"
    if item in TH_WORDS or first_word in TH_WORDS:
        return item, "th"
    if item in S_WORDS or first_word in S_WORDS:
        return item, "s"
    if item in Z_WORDS or first_word in Z_WORDS:
        return item, "z"
    if "susan" in item or "students" in item or "science" in item:
        return item, "mixed"
    if "zebra" in item or "puzzle" in item or "cheese" in item:
        return item, "z"
    return item, None


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    for feat in [zcr, centroid, rolloff, rms]:
        features.append(np.mean(feat))
        features.append(np.std(feat))

    return np.array(features).reshape(1, -1)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_confidence_level(confidence):
    if confidence >= 90:
        return "High confidence"
    if confidence >= 70:
        return "Medium confidence"
    return "Low confidence — review manually"


def get_practice_priority(prediction, confidence):
    prediction = str(prediction).lower()

    if confidence < 70:
        return "Review manually"
    if prediction == "unclear" and confidence >= 90:
        return "High priority"
    if prediction == "unclear":
        return "Medium priority"
    if prediction == "clear" and confidence >= 90:
        return "Low priority"
    return "Low / medium priority"


def make_reliable_feedback(prediction, confidence, target_sound, item_name):
    prediction = str(prediction).lower()

    if target_sound == "mixed":
        sound_text = "/s/ and /z/ sounds"
    elif target_sound:
        sound_text = f"/{target_sound}/ sound"
    else:
        sound_text = "target sound"

    item_text = f"for '{item_name}' " if item_name else ""

    if confidence < 70:
        return (
            f"The recording {item_text}was predicted as {prediction} with {confidence:.1f}% confidence, "
            f"which is a low-confidence result. Review this clip manually and keep practicing the {sound_text} slowly. "
            f"This is practice feedback only."
        )
    if prediction == "clear":
        return (
            f"The recording {item_text}was predicted as clear with {confidence:.1f}% confidence. "
            f"Keep practicing the {sound_text} slowly and consistently. "
            f"This is practice feedback only."
        )
    return (
        f"The recording {item_text}was predicted as unclear with {confidence:.1f}% confidence. "
        f"Practice it slowly and repeat it in short sets while focusing on the {sound_text}. "
        f"This is practice feedback only."
    )


def clean_llama_text(text):
    text = text.strip()
    bad_starts = [
        "here are two short sentences rewriting the original feedback:",
        "here are two short sentences:",
        "here are the two sentences:",
        "here is the rewritten feedback:",
        "rewritten feedback:",
        "original feedback:",
    ]
    lowered = text.lower()
    for phrase in bad_starts:
        if lowered.startswith(phrase):
            text = text[len(phrase):].strip()
            break
    return text


@st.cache_data(ttl=60)
def ollama_is_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        return response.status_code == 200
    except Exception:
        return False


def make_llama_feedback(reliable_feedback, prediction):
    if not ollama_is_available():
        return reliable_feedback, False

    prompt = f"""
You are writing user-facing feedback for a speech-practice website.

Rewrite the original feedback into exactly 2 short sentences.

Rules:
- Only output the final feedback.
- Do not introduce the response.
- Do not add new information.
- Do not contradict the prediction.
- If the prediction is clear, do not say the recording has problems, mistakes, rough sounds, or needs improvement.
- If the prediction is unclear, give one simple practice suggestion.
- Do not mention therapy, treatment, diagnosis, disorder, or speech-language pathologist.
- Do not use technical speech terms like alveolar ridge, guttural, placement, or articulation.
- Keep it simple, supportive, and non-clinical.
- The final sentence must include: "This is practice feedback only."

Prediction: {prediction}

Original feedback:
{reliable_feedback}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=45,
        )
        if response.status_code != 200:
            return reliable_feedback, False

        llama_text = clean_llama_text(response.json().get("response", "").strip())
        banned_phrases = [
            "here are",
            "rewriting",
            "original feedback",
            "therapy",
            "treatment",
            "diagnosis",
            "disorder",
            "speech-language pathologist",
            "alveolar ridge",
            "guttural",
            "articulation",
            "placement",
        ]
        contradiction_phrases_for_clear = [
            "room for improvement",
            "needs improvement",
            "need to work",
            "sounds rough",
            "rough",
            "tricky sounds",
            "mistake",
            "mistakes",
            "unclear",
        ]
        lower_text = llama_text.lower()

        if any(phrase in lower_text for phrase in banned_phrases):
            return reliable_feedback, False
        if str(prediction).lower() == "clear":
            if any(phrase in lower_text for phrase in contradiction_phrases_for_clear):
                return reliable_feedback, False
        if "practice feedback only" not in lower_text:
            llama_text += " This is practice feedback only."
        if llama_text:
            return llama_text, True
        return reliable_feedback, False
    except Exception:
        return reliable_feedback, False


def predict_from_audio_bytes(audio_bytes, suffix=".wav"):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        features = extract_features(tmp_path)
        model = load_model()
        prediction = model.predict(features)[0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)
            confidence = float(probabilities[classes.index(prediction)] * 100)
        else:
            confidence = 0.0
        return prediction, confidence
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


# -----------------------------
# Progress log
# -----------------------------

def load_progress_log():
    if st.session_state.progress_df is not None:
        return st.session_state.progress_df.copy()
    df = load_csv_if_exists(PROGRESS_LOG)
    if df is None:
        return pd.DataFrame()
    st.session_state.progress_df = df
    return df.copy()


def save_progress_log(row):
    current = load_progress_log()
    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    st.session_state.progress_df = updated
    try:
        updated.to_csv(PROGRESS_LOG, index=False)
    except Exception:
        pass
    return updated


# -----------------------------
# Practice helpers
# -----------------------------

def filter_practice_words(df, difficulty, search_text, extra_filters=None):
    filtered = df.copy()
    if extra_filters:
        for column, value in extra_filters.items():
            if value not in (None, "all") and column in filtered.columns:
                filtered = filtered[filtered[column] == value]
    if difficulty != "all":
        filtered = filtered[filtered["difficulty"] == difficulty]
    query = (search_text or "").strip().lower()
    if query:
        searchable = filtered["word"].astype(str).str.lower().str.contains(query, na=False)
        if "notes" in filtered.columns:
            searchable = searchable | filtered["notes"].astype(str).str.lower().str.contains(query, na=False)
        filtered = filtered[searchable]
    return filtered.reset_index(drop=True)


def filter_word_bank(df, sound, difficulty, search_text):
    return filter_practice_words(
        df,
        difficulty,
        search_text,
        extra_filters={"target_sound": sound},
    )


def friendly_practice_note(row, mode="sound"):
    notes = str(row.get("notes", "")).strip()
    word = str(row.get("word", "")).strip()
    difficulty = str(row.get("difficulty", "")).title()
    if mode == "alphabet":
        letter = str(row.get("letter", "")).upper()
        if not notes or notes.lower().startswith("alphabet practice") or notes.lower().startswith("matched cmu"):
            return f"Practice '{word}' slowly. This word starts with the letter {letter}. Difficulty: {difficulty or 'n/a'}."
        return notes
    sound = str(row.get("target_sound", "")).lower()
    sound_label = SOUND_LABELS.get(sound, sound)
    if not notes or notes.lower().startswith("matched cmu"):
        return f"Practice '{word}' slowly and keep a clear {sound_label} sound. Difficulty: {difficulty or 'n/a'}."
    return notes


def make_practice_plan(selected_words, focus_label, difficulty, practice_length, title="SpeakClear AI Practice Plan"):
    plan_lines = [
        title,
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Focus: {focus_label}",
        f"Difficulty filter: {difficulty}",
        f"Practice length: {practice_length}",
        "",
        "Safety note: SpeakClear AI is for non-clinical speech practice only. It is not a diagnosis or treatment tool.",
        "",
        "Words:",
    ]
    for word in selected_words:
        plan_lines.append(f"- {word}")
    plan_lines.append("")
    plan_lines.append("Steps:")

    steps = []
    for word in selected_words:
        steps.append(f"Say '{word}' slowly 3 times.")
        steps.append(f"Say '{word}' at normal speed 2 times.")
        steps.append(f"Record one version of '{word}' and analyze it in Try the Demo.")
    for index, step in enumerate(steps, start=1):
        plan_lines.append(f"{index}. {step}")

    plan_lines.extend(
        [
            "",
            "Recording tips:",
            "1. Use a quiet room.",
            "2. Keep the microphone distance consistent.",
            "3. Record one word or one short sentence at a time.",
            "4. WAV upload is the most reliable option.",
        ]
    )
    return steps, "\n".join(plan_lines)


def render_word_cards(words, max_cards=24):
    preview = list(words)[:max_cards]
    if not preview:
        return
    badge_cols = st.columns(4)
    for index, word in enumerate(preview):
        with badge_cols[index % 4]:
            st.markdown(f'<span class="badge">{escape(str(word))}</span>', unsafe_allow_html=True)
    if len(words) > max_cards:
        st.caption(f"Showing {max_cards} of {len(words)} matching words. The table below has the full list.")


def render_random_word_card(row, mode="sound"):
    word = escape(str(row.get("word", "")))
    difficulty = escape(str(row.get("difficulty", "")).title())
    note = escape(friendly_practice_note(row, mode=mode))
    if mode == "alphabet":
        focus_html = f"<p><strong>Letter:</strong> {escape(str(row.get('letter', '')).upper())}</p>"
    else:
        sound = str(row.get("target_sound", ""))
        focus_html = f"<p><strong>Sound:</strong> {escape(SOUND_LABELS.get(sound, sound))}</p>"
    st.markdown(
        f"""
        <div class="result-card">
            <p class="muted">Random practice word</p>
            <h3>{word}</h3>
            {focus_html}
            <p><strong>Difficulty:</strong> {difficulty}</p>
            <p>{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_practice_plan_state(prefix, selected_words, steps, plan_text, file_name):
    st.session_state[f"{prefix}_words"] = selected_words
    st.session_state[f"{prefix}_steps"] = steps
    st.session_state[f"{prefix}_plan_text"] = plan_text
    st.session_state[f"{prefix}_plan_name"] = file_name


def render_saved_practice_plan(prefix, download_label):
    steps = st.session_state.get(f"{prefix}_steps")
    if not steps:
        return
    st.markdown("#### Your practice items")
    st.write(", ".join(st.session_state.get(f"{prefix}_words", [])))
    st.markdown("#### Steps")
    for number, step in enumerate(steps, start=1):
        st.write(f"{number}. {step}")
    st.download_button(
        download_label,
        data=st.session_state.get(f"{prefix}_plan_text", ""),
        file_name=st.session_state.get(f"{prefix}_plan_name", "practice_plan.txt"),
        mime="text/plain",
        type="primary",
    )


def choose_practice_items(df, practice_length):
    words = df["word"].dropna().astype(str).tolist()
    if practice_length == "Quick practice":
        count = 3
    elif practice_length == "Normal practice":
        count = 5
    else:
        count = 8
    if not words:
        return []
    return random.sample(words, min(count, len(words)))


def audio_suffix(filename):
    suffix = Path(str(filename)).suffix.lower()
    if suffix in {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac"}:
        return suffix
    return ".wav"


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.markdown("### SpeakClear AI")
    st.caption("Accessible non-clinical speech-practice demo")

    nav_choice = st.radio(
        "Navigate",
        PAGES,
        key="nav_choice",
        help="Use arrow keys after focusing this list, or Tab to move to the next control.",
    )

    if nav_choice != st.session_state.current_page:
        st.session_state.current_page = nav_choice

    st.divider()
    st.subheader("Accessibility")
    st.checkbox(
        "Large text mode",
        key="large_text",
        help="Makes text larger across the app.",
    )
    st.checkbox(
        "High contrast mode",
        key="high_contrast",
        help="Uses stronger contrast for text, borders, and buttons.",
    )
    st.caption("Keyboard tip: Tab, Shift+Tab, and Enter work on most controls.")

    st.divider()
    st.markdown("**This app does not diagnose or treat speech differences.**")

apply_styles()

st.markdown(
    '<a class="skip-link" href="#main-content">Skip to main content</a>',
    unsafe_allow_html=True,
)
st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

page = st.session_state.current_page
word_bank_df, _ = get_word_bank_df()

if page == "Home":
    st.markdown(
        """
        <div class="app-hero">
            <div class="hero-kicker">Research prototype</div>
            <h1 class="hero-title">SpeakClear AI</h1>
            <p class="hero-subtitle">
                A personalized, accessible speech-practice demo. Upload a short recording,
                see a clear or unclear prediction, and get simple practice feedback.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="compact-header">
            <div class="hero-kicker">SpeakClear AI</div>
            <p class="hero-subtitle">Non-clinical speech-practice demo · {escape(page)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

safe_notice()

# -----------------------------
# Pages
# -----------------------------

if page == "Home":
    st.header("Welcome")
    st.write(
        "This app helps you practice speech sounds and alphabet words. It is a research demo, not a clinical tool."
    )

    start_col, demo_col = st.columns(2)
    with start_col:
        if st.button("Start Practicing", type="primary", use_container_width=True):
            go_to("Practice Studio")
    with demo_col:
        if st.button("Try a Recording", use_container_width=True):
            go_to("Try the Demo")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Personal audio clips", "245")
        st.caption("Real recorded clips used for model training and evaluation.")
    with m2:
        st.metric("Target speech sounds", "/s/, /z/, /r/, /th/")
        st.caption("Sounds used in the personalized ML dataset.")
    with m3:
        st.metric("Practice words", "3,000+")
        st.caption("Expanded sound and alphabet word banks for practice.")
    with m4:
        st.metric("Best model", "RF Classifier")
        st.caption("Random Forest model used for prediction.")
    st.caption(
        "The 245 personal audio clips are the recorded training and evaluation dataset. "
        "The 3,000+ practice words are a separate word bank for Sound Practice and Alphabet Practice. "
        "The model was not trained on the online practice word lists."
    )

    st.subheader("How to use this app")
    how_to_steps = [
        ("Pick a sound or letter", "Open Practice Studio and choose Sound Practice or Alphabet Practice."),
        ("Practice words", "Filter the word bank, try a random word, and download a short plan."),
        ("Record or upload audio", "Use a WAV file, or optionally record in the browser."),
        ("Get a clear/unclear prediction", "The RF Classifier estimates whether the clip sounds clear or unclear."),
        ("Review feedback and track progress", "Read the practice feedback and save attempts in Progress Tracker."),
    ]
    how_cols = st.columns(5)
    for index, (title, body) in enumerate(how_to_steps, start=1):
        with how_cols[index - 1]:
            st.markdown(
                f"""
                <div class="mini-card">
                    <strong>Step {index}</strong>
                    <h4>{escape(title)}</h4>
                    <p>{escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("What you can do")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="section-card">
                <h3>1. Practice</h3>
                <p>Practice /s/, /z/, /r/, and /th/, or choose alphabet words from A to Z.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Practice Studio", use_container_width=True):
            go_to("Practice Studio")
    with c2:
        st.markdown(
            """
            <div class="section-card">
                <h3>2. Analyze</h3>
                <p>Upload a WAV file, or optionally record in the browser, then see a clear or unclear prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Try the Demo", use_container_width=True):
            go_to("Try the Demo")
    with c3:
        st.markdown(
            """
            <div class="section-card">
                <h3>3. Review</h3>
                <p>Look at model results, graphs, and saved practice attempts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Research Results", use_container_width=True):
            go_to("Research Results")

    with st.expander("What happens behind the scenes"):
        st.write(
            "The app extracts acoustic features such as MFCCs and loudness, then a Random Forest "
            "classifier predicts clear or unclear pronunciation. You also see confidence, practice priority, "
            "and simple non-clinical feedback."
        )

    st.subheader("Practice options")
    sound_cols = st.columns(5)
    home_options = [
        ("s", SOUND_LABELS["s"], SOUND_DESCRIPTIONS["s"]),
        ("z", SOUND_LABELS["z"], SOUND_DESCRIPTIONS["z"]),
        ("r", SOUND_LABELS["r"], SOUND_DESCRIPTIONS["r"]),
        ("th", SOUND_LABELS["th"], SOUND_DESCRIPTIONS["th"]),
        ("alphabet", "A to Z", "Practice everyday words that start with each letter of the alphabet."),
    ]
    for index, (_key, title, body) in enumerate(home_options):
        with sound_cols[index]:
            st.markdown(
                f"""
                <div class="sound-card">
                    <strong>{escape(title)}</strong>
                    <p>{escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    info_box(
        "Privacy note",
        "Raw research recordings are not included in this public app. "
        "Any clip you analyze is processed to make a prediction and is not added to the project dataset.",
    )

elif page == "Practice Studio":
    st.header("Practice Studio")
    st.write(
        "Choose Sound Practice or Alphabet Practice. Filter words, try a random word, and download a practice plan."
    )

    practice_mode = st.radio(
        "Practice mode",
        ["Sound Practice", "Alphabet Practice"],
        horizontal=True,
        key="practice_mode",
        help="Sound Practice uses /s/, /z/, /r/, and /th/. Alphabet Practice uses words from A to Z.",
    )

    if practice_mode == "Sound Practice":
        sound_bank_df, sound_bank_from_csv = get_sound_word_bank()
        if sound_bank_from_csv:
            st.caption(f"Loaded sound word bank from `{WORD_BANK_PATH}` ({len(sound_bank_df)} words).")
        else:
            st.warning(
                f"Could not load `{WORD_BANK_PATH}`. Using the smaller built-in sound word lists instead."
            )

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            practice_sound = st.selectbox(
                "Target sound",
                SOUND_PRACTICE_SOUNDS,
                format_func=lambda value: SOUND_LABELS.get(value, value),
                help="Choose the sound you want to practice: /s/, /z/, /r/, or /th/.",
            )
        with filter_col2:
            difficulty_filter = st.selectbox(
                "Difficulty",
                ["all", "easy", "medium", "hard"],
                format_func=lambda value: "All difficulties" if value == "all" else value.title(),
                help="Filter words by easy, medium, or hard practice level.",
                key="sound_difficulty",
            )
        with filter_col3:
            practice_length = st.selectbox(
                "Practice length",
                ["Quick practice", "Normal practice", "Long practice"],
                help="Quick uses 3 words, normal uses 5, and long uses up to 8.",
                key="sound_length",
            )

        search_text = st.text_input(
            "Search sound-practice words",
            placeholder="Try sun, zero, right, or thank",
            help="Type part of a word to narrow the list.",
            key="sound_search",
        )

        st.subheader("Sound description")
        st.info(SOUND_DESCRIPTIONS[practice_sound])

        filtered_bank = filter_practice_words(
            sound_bank_df,
            difficulty_filter,
            search_text,
            extra_filters={"target_sound": practice_sound},
        )

        st.subheader("Sound word bank")
        if filtered_bank.empty:
            st.warning("No words match these filters. Try another sound, difficulty, or search term.")
        else:
            count_col1, count_col2, count_col3 = st.columns(3)
            count_col1.metric("Matching words", len(filtered_bank))
            count_col2.metric("Easy", int((filtered_bank["difficulty"] == "easy").sum()))
            count_col3.metric("Hard", int((filtered_bank["difficulty"] == "hard").sum()))
            render_word_cards(filtered_bank["word"].tolist())
            display_bank = filtered_bank.copy()
            display_bank["target_sound"] = display_bank["target_sound"].map(
                lambda value: SOUND_LABELS.get(value, value)
            )
            display_bank["notes"] = [
                friendly_practice_note(row, mode="sound")
                for row in filtered_bank.to_dict("records")
            ]
            st.dataframe(
                display_bank.rename(
                    columns={
                        "word": "Word",
                        "target_sound": "Target sound",
                        "source": "Source",
                        "difficulty": "Difficulty",
                        "notes": "Practice note",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Random practice word")
        random_col1, random_col2 = st.columns([1, 2])
        with random_col1:
            if st.button("Random Practice Word", type="primary", use_container_width=True):
                if filtered_bank.empty:
                    st.session_state.pop("sound_challenge_row", None)
                    st.warning("No matching words are available right now.")
                else:
                    st.session_state.sound_challenge_row = filtered_bank.sample(1).iloc[0].to_dict()
        if st.session_state.get("sound_challenge_row"):
            with random_col2:
                render_random_word_card(st.session_state.sound_challenge_row, mode="sound")
            st.write("Suggested practice: say it slowly 3 times, then at normal speed 2 times, then record one version.")

        st.subheader("Sound practice plan")
        if st.button("Generate Practice Plan"):
            selected_words = choose_practice_items(filtered_bank, practice_length)
            if not selected_words:
                st.warning("No matching words are available for a practice plan.")
            else:
                steps, plan_text = make_practice_plan(
                    selected_words,
                    SOUND_LABELS.get(practice_sound, practice_sound),
                    difficulty_filter,
                    practice_length,
                    title="SpeakClear AI Sound Practice Plan",
                )
                save_practice_plan_state(
                    "sound",
                    selected_words,
                    steps,
                    plan_text,
                    f"practice_plan_{practice_sound}_{difficulty_filter}.txt",
                )
        render_saved_practice_plan("sound", "Download practice plan")

        st.subheader("Personal practice priorities from the research set")
        feedback_df = load_csv_if_exists(RESULTS_DIR / "practice_feedback_by_item.csv")
        if feedback_df is not None and "target_sound" in feedback_df.columns:
            filtered_feedback = feedback_df[
                feedback_df["target_sound"].astype(str).str.lower() == practice_sound
            ]
            if filtered_feedback.empty:
                st.info("No saved priority items were found for this sound.")
            else:
                show_cols = [
                    col
                    for col in ["word", "target_sound", "unclear_rate", "priority", "feedback"]
                    if col in filtered_feedback.columns
                ]
                display_feedback = filtered_feedback[show_cols].head(10).copy()
                if "unclear_rate" in display_feedback.columns:
                    display_feedback["unclear_rate"] = (
                        display_feedback["unclear_rate"].astype(float) * 100
                    ).round(0).astype(int).astype(str) + "%"
                st.dataframe(display_feedback, use_container_width=True, hide_index=True)
        else:
            st.info("No practice priority file was found yet.")

    else:
        alphabet_bank_df, alphabet_bank_from_csv = get_alphabet_word_bank()
        if alphabet_bank_from_csv:
            st.caption(
                f"Loaded alphabet word bank from `{ALPHABET_WORD_BANK_PATH}` ({len(alphabet_bank_df)} words)."
            )
        else:
            st.warning(
                f"`{ALPHABET_WORD_BANK_PATH}` is missing or could not be read. "
                "Alphabet Practice is using a small built-in A–Z fallback list instead. "
                "Add that CSV to enable the full alphabet word bank."
            )

        st.info("Alphabet Practice helps you practice everyday words from A to Z. This is still non-clinical practice only.")

        alpha_col1, alpha_col2, alpha_col3 = st.columns(3)
        with alpha_col1:
            practice_letter = st.selectbox(
                "Letter",
                ALPHABET_LETTERS,
                help="Choose a letter from A to Z.",
            )
        with alpha_col2:
            alpha_difficulty = st.selectbox(
                "Difficulty",
                ["all", "easy", "medium", "hard"],
                format_func=lambda value: "All difficulties" if value == "all" else value.title(),
                help="Filter alphabet words by easy, medium, or hard practice level.",
                key="alpha_difficulty",
            )
        with alpha_col3:
            alpha_length = st.selectbox(
                "Practice length",
                ["Quick practice", "Normal practice", "Long practice"],
                help="Quick uses 3 words, normal uses 5, and long uses up to 8.",
                key="alpha_length",
            )

        alpha_search = st.text_input(
            "Search alphabet words",
            placeholder="Try apple, zebra, or queen",
            help="Type part of a word to narrow the alphabet list.",
            key="alpha_search",
        )

        filtered_alpha = filter_practice_words(
            alphabet_bank_df,
            alpha_difficulty,
            alpha_search,
            extra_filters={"letter": practice_letter},
        )

        st.subheader(f"Alphabet word bank for {practice_letter}")
        if filtered_alpha.empty:
            st.warning("No words match these filters. Try another letter, difficulty, or search term.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Matching words", len(filtered_alpha))
            a2.metric("Letter", practice_letter)
            a3.metric("Hard", int((filtered_alpha["difficulty"] == "hard").sum()))
            render_word_cards(filtered_alpha["word"].tolist())
            display_alpha = filtered_alpha.copy()
            display_alpha["notes"] = [
                friendly_practice_note(row, mode="alphabet")
                for row in filtered_alpha.to_dict("records")
            ]
            st.dataframe(
                display_alpha.rename(
                    columns={
                        "word": "Word",
                        "letter": "Letter",
                        "source": "Source",
                        "difficulty": "Difficulty",
                        "notes": "Practice note",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Random alphabet word")
        alpha_random1, alpha_random2 = st.columns([1, 2])
        with alpha_random1:
            if st.button("Random Alphabet Word", type="primary", use_container_width=True):
                if filtered_alpha.empty:
                    st.session_state.pop("alpha_challenge_row", None)
                    st.warning("No matching alphabet words are available right now.")
                else:
                    st.session_state.alpha_challenge_row = filtered_alpha.sample(1).iloc[0].to_dict()
        if st.session_state.get("alpha_challenge_row"):
            with alpha_random2:
                render_random_word_card(st.session_state.alpha_challenge_row, mode="alphabet")
            st.write("Suggested practice: say it slowly 3 times, then at normal speed 2 times, then record one version.")

        st.subheader("Alphabet practice plan")
        if st.button("Generate Alphabet Practice Plan"):
            selected_words = choose_practice_items(filtered_alpha, alpha_length)
            if not selected_words:
                st.warning("No matching alphabet words are available for a practice plan.")
            else:
                steps, plan_text = make_practice_plan(
                    selected_words,
                    f"Letter {practice_letter}",
                    alpha_difficulty,
                    alpha_length,
                    title="SpeakClear AI Alphabet Practice Plan",
                )
                save_practice_plan_state(
                    "alpha",
                    selected_words,
                    steps,
                    plan_text,
                    f"alphabet_practice_plan_{practice_letter}_{alpha_difficulty}.txt",
                )
        render_saved_practice_plan("alpha", "Download alphabet practice plan")

    st.subheader("Recording tips")
    st.write(
        """
1. Record in a quiet room.
2. Keep the microphone distance consistent.
3. Use one word or one short sentence per recording.
4. WAV upload is the main reliable option.
5. Helpful filenames look like `bath-1.wav`, `right-3.wav`, or `zero-2.wav`.
        """
    )

elif page == "Try the Demo":
    st.header("Try the Demo")
    st.write(
        "Upload a short WAV recording to get a clear or unclear prediction. "
        "Browser recording is available as an extra option."
    )

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python3 train_save_model.py` first.")
        st.stop()

    try:
        load_model()
    except Exception:
        st.error("The model file could not be loaded. Please check `models/speakclear_random_forest.joblib`.")
        st.stop()

    selected_sound = st.selectbox(
        "Target sound you practiced",
        ["s", "z", "r", "th", "mixed"],
        format_func=lambda value: SOUND_LABELS.get(value, value),
        help="Choose the sound you were trying to practice.",
    )
    spoken_item = st.text_input(
        "Word or sentence you said (optional)",
        placeholder="Example: thank, zero, or susan saw the zebra at the zoo",
        help="If you type the word, feedback can mention it. This is especially helpful for browser recordings.",
    )

    input_method = st.radio(
        "How do you want to add audio?",
        ["Upload a WAV file (recommended)", "Record in browser (optional)"],
        help="WAV upload is the most reliable option. Browser recording depends on your browser and microphone permission.",
    )

    audio_bytes = None
    source_name = None
    audio_label = None

    if input_method.startswith("Upload"):
        uploaded_file = st.file_uploader(
            "Upload a WAV audio recording",
            type=["wav"],
            help="Upload a short WAV file, ideally one word or one short sentence.",
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name
            audio_label = uploaded_file.name
            st.audio(audio_bytes, format="audio/wav")
    else:
        if hasattr(st, "audio_input"):
            audio_input_kwargs = {
                "label": "Record a short practice clip",
                "help": "Allow microphone access, record one word or short sentence, then stop.",
            }
            try:
                recorded_file = st.audio_input(sample_rate=16000, **audio_input_kwargs)
            except TypeError:
                recorded_file = st.audio_input(**audio_input_kwargs)
            if recorded_file is not None:
                audio_bytes = recorded_file.getvalue()
                source_name = getattr(recorded_file, "name", "browser_recording.wav") or "browser_recording.wav"
                audio_label = "Browser recording"
                st.audio(audio_bytes)
                st.caption("Browser recording can vary by device. If analysis fails, upload a WAV file instead.")
        else:
            st.warning(
                "Browser recording is not available in this Streamlit version. Please upload a WAV file instead."
            )

    if audio_bytes is not None:
        item_name, inferred_sound = infer_target_sound(source_name, word_bank_df)
        if spoken_item.strip():
            typed_name = spoken_item.strip().lower()
            typed_item, typed_sound = infer_target_sound(f"{typed_name}.wav", word_bank_df)
            item_name = typed_item or typed_name
            if typed_sound:
                inferred_sound = typed_sound

        if not item_name or item_name in {"browser recording", "recording", "audio"}:
            item_name = spoken_item.strip().lower() or "this recording"

        target_sound = selected_sound
        info1, info2, info3 = st.columns(3)
        info1.metric("Item", item_name)
        info2.metric("Detected sound", SOUND_LABELS.get(inferred_sound, "Unknown"))
        info3.metric("Selected sound", SOUND_LABELS.get(selected_sound, selected_sound))

        if inferred_sound is not None and inferred_sound != selected_sound:
            st.warning(
                f"This clip looks like {SOUND_LABELS.get(inferred_sound, inferred_sound)}, "
                f"but you selected {SOUND_LABELS.get(selected_sound, selected_sound)}. "
                f"Feedback will use the detected sound: {SOUND_LABELS.get(inferred_sound, inferred_sound)}."
            )
            target_sound = inferred_sound

        try:
            with st.spinner("Analyzing audio..."):
                prediction, confidence = predict_from_audio_bytes(
                    audio_bytes,
                    suffix=audio_suffix(source_name),
                )
        except Exception:
            st.error(
                "The audio could not be analyzed. Please try a short WAV file. "
                "Browser recordings do not always work the same way on every device."
            )
            st.stop()

        confidence_level = get_confidence_level(confidence)
        practice_priority = get_practice_priority(prediction, confidence)
        reliable_feedback = make_reliable_feedback(
            prediction,
            confidence,
            target_sound,
            item_name if item_name != "this recording" else "",
        )
        llama_feedback, llama_used = make_llama_feedback(reliable_feedback, prediction)

        st.subheader("Result")
        result_banner(prediction)

        result1, result2, result3 = st.columns(3)
        with result1:
            st.metric("Prediction label", str(prediction).title())
        with result2:
            st.metric("Confidence", f"{confidence:.1f}%")
            st.progress(min(max(confidence / 100.0, 0.0), 1.0))
            st.caption(confidence_level)
        with result3:
            st.metric("Practice priority", practice_priority)
            st.caption("Priority is based on the prediction and confidence, not color alone.")

        st.subheader("Reliable practice feedback")
        st.write(reliable_feedback)
        st.caption("This rule-based feedback is always available, including on Streamlit Community Cloud.")

        st.subheader("AI-rewritten feedback")
        if llama_used:
            st.success("Local Llama rewrite was used. Safety checks were applied.")
        else:
            st.info(
                "Local Ollama / Llama is not available here, so the app is using the reliable rule-based feedback. "
                "This is expected on Streamlit Community Cloud."
            )
        st.write(llama_feedback)

        feedback_report = f"""SpeakClear AI Feedback Report

Audio source: {audio_label}
Item: {item_name}
Target sound: {target_sound}
Prediction: {prediction}
Confidence: {confidence:.1f}%
Confidence level: {confidence_level}
Practice priority: {practice_priority}
Feedback source: {"Local Llama rewrite" if llama_used else "Rule-based fallback"}

Reliable feedback:
{reliable_feedback}

AI-rewritten feedback:
{llama_feedback}

Disclaimer:
This is practice feedback only. It is not a diagnosis or treatment tool.
"""

        download_col, save_col = st.columns(2)
        with download_col:
            st.download_button(
                "Download feedback report",
                data=feedback_report,
                file_name=f"feedback_{str(item_name).replace(' ', '_')}.txt",
                mime="text/plain",
            )
        with save_col:
            if st.button("Save attempt to Progress Tracker", type="primary"):
                save_progress_log(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file_name": audio_label,
                        "item_name": item_name,
                        "target_sound": target_sound,
                        "prediction": prediction,
                        "confidence": round(confidence, 1),
                        "confidence_level": confidence_level,
                        "practice_priority": practice_priority,
                        "reliable_feedback": reliable_feedback,
                        "llm_feedback": llama_feedback,
                        "feedback_source": "llama" if llama_used else "rule_based",
                    }
                )
                st.success("Saved to Progress Tracker.")

        with st.expander("Technical details"):
            st.write(
                "This prediction uses MFCCs, zero-crossing rate, spectral centroid, "
                "spectral rolloff, and RMS energy."
            )
            st.write("The model was trained on a personal 245-clip speech-practice dataset.")
            st.write(
                "The optional LLM layer only rewrites safe rule-based feedback. "
                "If Ollama is unavailable or the rewrite fails a safety check, the rule-based text is shown instead."
            )

elif page == "Progress Tracker":
    st.header("Progress Tracker")
    st.write("Saved attempts stay in this app session and in `practice_progress_log.csv` when the server can write files.")

    progress_df = load_progress_log()
    if progress_df.empty:
        st.info("No saved attempts yet. Analyze a recording in Try the Demo, then click Save attempt.")
        if st.button("Go to Try the Demo"):
            go_to("Try the Demo")
    else:
        total_attempts = len(progress_df)
        pred_series = progress_df["prediction"].astype(str).str.lower() if "prediction" in progress_df.columns else pd.Series(dtype=str)
        clear_count = int((pred_series == "clear").sum())
        unclear_count = int((pred_series == "unclear").sum())
        avg_confidence = (
            pd.to_numeric(progress_df["confidence"], errors="coerce").mean()
            if "confidence" in progress_df.columns
            else None
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Saved attempts", total_attempts)
        c2.metric("Clear", clear_count)
        c3.metric("Unclear", unclear_count)
        c4.metric(
            "Average confidence",
            f"{avg_confidence:.1f}%" if pd.notna(avg_confidence) else "n/a",
        )

        st.subheader("Practice history")
        st.dataframe(progress_df, use_container_width=True, hide_index=True)

        chart1, chart2 = st.columns(2)
        with chart1:
            st.subheader("Attempts by target sound")
            if "target_sound" in progress_df.columns:
                sound_counts = (
                    progress_df["target_sound"].astype(str).value_counts().rename_axis("Target sound").reset_index(name="Attempts")
                )
                sound_counts["Target sound"] = sound_counts["Target sound"].map(
                    lambda value: SOUND_LABELS.get(value, value)
                )
                st.bar_chart(sound_counts, x="Target sound", y="Attempts")
                st.caption("This chart shows how many saved attempts you have for each target sound.")
            else:
                st.info("No target-sound column is available in the progress log.")
        with chart2:
            st.subheader("Clear vs unclear")
            if "prediction" in progress_df.columns:
                pred_counts = (
                    progress_df["prediction"].astype(str).str.title().value_counts().rename_axis("Prediction").reset_index(name="Attempts")
                )
                st.bar_chart(pred_counts, x="Prediction", y="Attempts")
                st.caption("This chart uses text labels for Clear and Unclear, not color alone.")
            else:
                st.info("No prediction column is available in the progress log.")

        if "target_sound" in progress_df.columns and "prediction" in progress_df.columns:
            st.subheader("Breakdown by sound and prediction")
            breakdown = (
                progress_df.assign(
                    target_sound=progress_df["target_sound"].astype(str),
                    prediction=progress_df["prediction"].astype(str).str.title(),
                )
                .groupby(["target_sound", "prediction"], dropna=False)
                .size()
                .reset_index(name="attempts")
            )
            breakdown["target_sound"] = breakdown["target_sound"].map(
                lambda value: SOUND_LABELS.get(value, value)
            )
            st.dataframe(
                breakdown.rename(
                    columns={
                        "target_sound": "Target sound",
                        "prediction": "Prediction",
                        "attempts": "Attempts",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download progress log",
            data=progress_df.to_csv(index=False),
            file_name="practice_progress_log.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(
            "On Streamlit Community Cloud, saved files may reset when the app restarts. Download your log if you want to keep it."
        )

elif page == "Research Results":
    st.header("Research Results")
    st.write(
        "These results come from a personal 245-clip speech-practice dataset. "
        "They describe this prototype, not clinical performance."
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total clips", "245")
    r2.metric("Clear clips", "188")
    r3.metric("Unclear clips", "57")
    r4.metric("Best model", "RF Classifier")

    st.subheader("What the results mean")
    st.markdown(
        """
        <div class="info-box">
            <p>Accuracy alone was misleading in this project.</p>
            <ul>
                <li>Most clips were already labeled clear (188 clear vs 57 unclear).</li>
                <li>The Always Clear baseline had high accuracy (77.4%) because it always guessed the common label.</li>
                <li>That baseline never found unclear speech, so its unclear F1 score was 0.00.</li>
                <li>Random Forest was more useful because it detected unclear clips better, with an unclear F1 score of 0.48.</li>
                <li>This is one-speaker research. It is not a diagnosis or treatment tool.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model comparison")
    model_df = load_csv_if_exists(RESULTS_DIR / "model_results.csv")
    if model_df is not None:
        pretty = model_df.copy()
        percent_cols = [col for col in ["accuracy", "precision_unclear", "recall_unclear"] if col in pretty.columns]
        for col in percent_cols:
            pretty[col] = (pd.to_numeric(pretty[col], errors="coerce") * 100).round(1).astype(str) + "%"
        if "f1_unclear" in pretty.columns:
            pretty["f1_unclear"] = pd.to_numeric(pretty["f1_unclear"], errors="coerce").round(2)
        pretty = pretty.rename(
            columns={
                "model": "Model",
                "accuracy": "Accuracy",
                "precision_unclear": "Unclear precision",
                "recall_unclear": "Unclear recall",
                "f1_unclear": "Unclear F1",
            }
        )
        st.dataframe(pretty, use_container_width=True, hide_index=True)
        st.caption("Unclear F1 is the most important score here. Accuracy alone is not enough.")
    else:
        st.info("`results/model_results.csv` was not found.")

    st.write(
        "Random Forest reached 75.8% accuracy and an unclear-class F1 score of 0.48. "
        "The Always Clear baseline reached 77.4% accuracy, but its unclear F1 score was 0.00. "
        "A model can look accurate while missing every unclear clip. Random Forest was chosen because it was better at finding unclear practice items, not because it had the highest accuracy."
    )

    st.subheader("Confusion matrices")
    matrix_cols = st.columns(3)
    matrices = [
        ("always_clear_baseline_confusion_matrix.png", "Always Clear baseline"),
        ("logistic_regression_confusion_matrix.png", "Logistic Regression"),
        ("random_forest_confusion_matrix.png", "Random Forest"),
    ]
    for column, (filename, caption) in zip(matrix_cols, matrices):
        path = RESULTS_DIR / filename
        with column:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{filename} was not found.")
    st.caption(
        "Each matrix shows correct and incorrect guesses. Random Forest is the only model shown here that catches a meaningful number of unclear clips."
    )

    st.subheader("Pattern analysis")
    pattern_tabs = st.tabs(["By sound", "By clip type", "Hardest items"])

    with pattern_tabs[0]:
        sound_graph = RESULTS_DIR / "unclear_rate_by_sound.png"
        sound_table = load_csv_if_exists(RESULTS_DIR / "unclear_rate_by_sound.csv")
        if sound_graph.exists():
            st.image(
                str(sound_graph),
                caption="Unclear rate by target sound. Higher bars mean that sound was harder in this dataset.",
                use_container_width=True,
            )
        if sound_table is not None:
            pretty_sound = sound_table.copy()
            if "unclear_rate" in pretty_sound.columns:
                pretty_sound["unclear_rate"] = (pd.to_numeric(pretty_sound["unclear_rate"], errors="coerce") * 100).round(1).astype(str) + "%"
            st.dataframe(pretty_sound, use_container_width=True, hide_index=True)
        st.write(
            "In this personal dataset, /th/ was the hardest target sound (40% unclear), "
            "followed by mixed /s/ and /z/ sentences. /r/ was the easiest overall."
        )

    with pattern_tabs[1]:
        clip_graph = RESULTS_DIR / "unclear_rate_by_clip_type.png"
        clip_table = load_csv_if_exists(RESULTS_DIR / "unclear_rate_by_clip_type.csv")
        if clip_graph.exists():
            st.image(
                str(clip_graph),
                caption="Unclear rate by clip type. Words and sentences are shown separately.",
                use_container_width=True,
            )
        if clip_table is not None:
            pretty_clip = clip_table.copy()
            if "unclear_rate" in pretty_clip.columns:
                pretty_clip["unclear_rate"] = (pd.to_numeric(pretty_clip["unclear_rate"], errors="coerce") * 100).round(1).astype(str) + "%"
            st.dataframe(pretty_clip, use_container_width=True, hide_index=True)
        st.write(
            "Words and sentences had similar unclear rates. Sentences were not automatically much harder than single words in this set."
        )

    with pattern_tabs[2]:
        top_graph = RESULTS_DIR / "top10_unclear_items.png"
        item_table = load_csv_if_exists(RESULTS_DIR / "unclear_rate_by_item.csv")
        if top_graph.exists():
            st.image(
                str(top_graph),
                caption="Top 10 hardest words and sentences from the research dataset.",
                use_container_width=True,
            )
        if item_table is not None:
            pretty_items = item_table.copy()
            if "unclear_rate" in pretty_items.columns:
                pretty_items["unclear_rate"] = (pd.to_numeric(pretty_items["unclear_rate"], errors="coerce") * 100).round(1).astype(str) + "%"
            st.dataframe(pretty_items.head(10), use_container_width=True, hide_index=True)
        st.write(
            "Items such as thank, right, sun, thirty, and 'zach said science was easy' were among the highest-priority practice targets."
        )

    st.subheader("LLM feedback evaluation")
    llama_summary = load_csv_if_exists(RESULTS_DIR / "llama_feedback_evaluation_summary.csv")
    if llama_summary is not None:
        pretty_llama = llama_summary.copy()
        pretty_llama = pretty_llama.rename(columns={"metric": "Rubric category", "average_score": "Average score"})
        st.dataframe(pretty_llama, use_container_width=True, hide_index=True)
    else:
        st.info("`results/llama_feedback_evaluation_summary.csv` was not found.")

    st.markdown(
        """
        The LLM rewrite layer was scored on 10 examples using a 10-point rubric:

        - **Matches prediction (1.5 / 2):** The rewrite usually followed the model, but sometimes added extra doubt.
        - **Safety (1.9 / 2):** This was the strongest category. The text almost always stayed non-clinical.
        - **Usefulness (1.4 / 2):** This was the weakest category. Advice was sometimes too general.
        - **Clarity (1.7 / 2):** Most rewrites were easy to read.
        - **Tone (1.8 / 2):** The tone was usually supportive.
        - **Total (8.2 / 10):** Helpful as a writing layer, but not a replacement for the rule-based feedback.
        """
    )
    st.write(
        "Because hosted deployment cannot rely on local Ollama, the app always keeps the rule-based feedback "
        "and only uses Llama when it is available and passes safety checks."
    )

    llama_examples = load_csv_if_exists(RESULTS_DIR / "llama_feedback_evaluation.csv")
    if llama_examples is None:
        llama_examples = load_csv_if_exists(Path("llama_feedback_evaluation.csv"))
    if llama_examples is not None:
        with st.expander("View scored feedback examples"):
            st.dataframe(llama_examples, use_container_width=True, hide_index=True)

elif page == "About & Limitations":
    st.header("About & Limitations")

    st.subheader("Research question")
    st.write(
        "Can machine learning analyze personalized speech-pronunciation patterns and generate useful "
        "non-clinical feedback for speech practice?"
    )

    st.subheader("Hypothesis")
    st.write(
        "If acoustic features from personalized speech-practice recordings are used to train a "
        "machine-learning classifier, then the system will be able to identify patterns in clear and "
        "unclear pronunciation clips and generate useful non-clinical feedback for practice."
    )

    st.subheader("Accessibility design choices")
    st.write(
        """
- Large text mode
- High contrast mode
- Clear headings and simple language
- Labels on every input
- Keyboard-friendly Streamlit controls
- Visible focus outlines
- No color-only meaning: predictions also use text labels
- Downloadable plans and reports for offline review
- Safety warning on every page
- Short feedback instead of long paragraphs
        """
    )

    st.subheader("Limitations")
    st.write(
        """
- The dataset contains recordings from one speaker, so results may not generalize to other speakers.
- Labels were assigned manually and may be subjective.
- The model is not clinically validated.
- The app should not be used for diagnosis or treatment.
- Browser recording quality depends on the device and browser.
- The optional LLM rewrite can be vague or inconsistent, so the app always keeps a rule-based fallback.
- Future work could include more speakers, expert label review, and deeper audio models such as Wav2Vec2.
        """
    )

    st.subheader("Technology used")
    st.write(
        """
- Python
- Streamlit
- librosa
- scikit-learn Random Forest
- pandas
- Optional local Ollama / Llama, with rule-based fallback
        """
    )

st.markdown(
    """
    <div class="footer-note">
        SpeakClear AI is an independent research prototype for non-clinical speech practice.
        It should not be used as a medical, diagnostic, or therapeutic tool.
    </div>
    """,
    unsafe_allow_html=True,
)
