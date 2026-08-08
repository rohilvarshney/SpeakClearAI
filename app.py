from pathlib import Path
import tempfile
import requests
import random
from datetime import datetime

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = Path("models/speakclear_random_forest.joblib")
RESULTS_DIR = Path("results")
PROGRESS_LOG = Path("practice_progress_log.csv")

st.set_page_config(
    page_title="SpeakClear AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Data
# -----------------------------

R_WORDS = [
    "red", "race", "right", "river", "around",
    "carrot", "mirror", "story", "car", "far"
]

TH_WORDS = [
    "think", "thin", "three", "thirty", "thank",
    "birthday", "nothing", "healthy", "bath", "teeth"
]

S_WORDS = [
    "sun", "seal", "seven", "sister", "science",
    "soup", "same", "bus", "yes", "class"
]

Z_WORDS = [
    "zoo", "zero", "zebra", "zipper", "zone",
    "buzz", "lazy", "puzzle", "music", "cheese"
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
    "s": "Practice steady /s/ sounds in words like sun, seal, and science.",
    "z": "Practice voiced /z/ sounds in words like zero, zoo, and buzz.",
    "r": "Practice /r/ sounds in words like right, river, mirror, and story.",
    "th": "Practice /th/ sounds in words like think, thank, bath, and teeth.",
    "mixed": "Practice sentence-level /s/ and /z/ combinations.",
}

# -----------------------------
# Accessibility and Styling
# -----------------------------

if "large_text" not in st.session_state:
    st.session_state["large_text"] = False

if "high_contrast" not in st.session_state:
    st.session_state["high_contrast"] = False


def apply_styles():
    large_text = st.session_state["large_text"]
    high_contrast = st.session_state["high_contrast"]

    base_font_size = "20px" if large_text else "17px"
    small_font_size = "17px" if large_text else "15px"

    if high_contrast:
        bg = "#050505"
        panel = "#111111"
        text = "#ffffff"
        muted = "#e8e8e8"
        border = "#ffffff"
        accent = "#ffdd00"
        soft = "#1d1d1d"
        success_bg = "#003b18"
        warning_bg = "#3d3200"
        error_bg = "#420000"
    else:
        bg = "#f7f9fc"
        panel = "#ffffff"
        text = "#111827"
        muted = "#4b5563"
        border = "#d7dde8"
        accent = "#315efb"
        soft = "#eef3ff"
        success_bg = "#e8f7ef"
        warning_bg = "#fff7dc"
        error_bg = "#ffe8e8"

    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-size: {base_font_size};
        }}

        .stApp {{
            background: {bg};
            color: {text};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {text};
            letter-spacing: -0.02em;
        }}

        p, li, label, span, div {{
            color: {text};
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }}

        section[data-testid="stSidebar"] {{
            background: {panel};
            border-right: 1px solid {border};
        }}

        .app-hero {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        }}

        .hero-title {{
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 8px;
            color: {text};
        }}

        .hero-subtitle {{
            font-size: 1.25rem;
            line-height: 1.6;
            color: {muted};
            max-width: 900px;
        }}

        .section-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        }}

        .mini-card {{
            background: {soft};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 14px;
            min-height: 140px;
        }}

        .sound-card {{
            background: {panel};
            border: 2px solid {border};
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 16px;
        }}

        .sound-card strong {{
            font-size: 1.5rem;
            color: {accent};
        }}

        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: {small_font_size};
            font-weight: 700;
            border: 1px solid {border};
            background: {soft};
            color: {text};
            margin-right: 8px;
            margin-bottom: 8px;
        }}

        .safe-box {{
            background: {warning_bg};
            border-left: 8px solid {accent};
            border-radius: 16px;
            padding: 18px;
            margin: 18px 0;
        }}

        .success-box {{
            background: {success_bg};
            border-left: 8px solid #22c55e;
            border-radius: 16px;
            padding: 18px;
            margin: 18px 0;
        }}

        .error-box {{
            background: {error_bg};
            border-left: 8px solid #ef4444;
            border-radius: 16px;
            padding: 18px;
            margin: 18px 0;
        }}

        .muted {{
            color: {muted};
            font-size: {small_font_size};
        }}

        .big-number {{
            font-size: 2.2rem;
            font-weight: 800;
            color: {accent};
        }}

        .footer-note {{
            color: {muted};
            font-size: {small_font_size};
            margin-top: 32px;
            border-top: 1px solid {border};
            padding-top: 18px;
        }}

        button[kind="primary"], .stButton button {{
            min-height: 44px;
            border-radius: 12px;
            font-weight: 700;
        }}

        div[data-testid="stMetric"] {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 18px;
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
        </style>
        """,
        unsafe_allow_html=True
    )


def card(title, body):
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_notice():
    st.markdown(
        """
        <div class="safe-box">
            <strong>Safety note:</strong> SpeakClear AI is for non-clinical speech practice only.
            It is not a diagnosis, treatment tool, or replacement for a speech-language pathologist.
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Core Functions
# -----------------------------

def normalize_uploaded_name(filename):
    stem = Path(filename).stem.lower()

    if "-" in stem:
        item = stem.rsplit("-", 1)[0]
    else:
        item = stem

    return item.replace("_", " ").strip()


def infer_target_sound(filename):
    item = normalize_uploaded_name(filename)
    first_word = item.split()[0] if item.split() else item

    if item in SENTENCE_TARGETS:
        return item, SENTENCE_TARGETS[item]

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
    else:
        sound_text = f"/{target_sound}/ sound"

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
        "original feedback:"
    ]

    lowered = text.lower()

    for phrase in bad_starts:
        if lowered.startswith(phrase):
            text = text[len(phrase):].strip()
            break

    return text


def make_llama_feedback(reliable_feedback, prediction):
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
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=45
        )

        if response.status_code == 200:
            llama_text = response.json().get("response", "").strip()
            llama_text = clean_llama_text(llama_text)

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
                "placement"
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
                "unclear"
            ]

            lower_text = llama_text.lower()

            if any(phrase in lower_text for phrase in banned_phrases):
                return reliable_feedback

            if str(prediction).lower() == "clear":
                if any(phrase in lower_text for phrase in contradiction_phrases_for_clear):
                    return reliable_feedback

            if "practice feedback only" not in lower_text:
                llama_text += " This is practice feedback only."

            if llama_text:
                return llama_text

        return reliable_feedback

    except Exception:
        return reliable_feedback


def save_progress_log(row):
    row_df = pd.DataFrame([row])

    if PROGRESS_LOG.exists():
        old_df = pd.read_csv(PROGRESS_LOG)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df

    new_df.to_csv(PROGRESS_LOG, index=False)


def load_progress_log():
    if PROGRESS_LOG.exists():
        return pd.read_csv(PROGRESS_LOG)
    return pd.DataFrame()


def make_practice_plan(target_sound, difficulty):
    words = WORD_BANK[target_sound]

    if difficulty == "Quick practice":
        count = 3
    elif difficulty == "Normal practice":
        count = 5
    else:
        count = min(8, len(words))

    selected = random.sample(words, min(count, len(words)))

    plan = []
    for word in selected:
        plan.append(f"Say '{word}' slowly 3 times.")
        plan.append(f"Say '{word}' at normal speed 2 times.")
        plan.append(f"Record one version of '{word}' and upload it in Try the Demo.")

    return selected, plan


def load_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    return None


# -----------------------------
# Sidebar Controls
# -----------------------------

with st.sidebar:
    st.title("🎙️ SpeakClear AI")
    st.caption("Accessible speech-practice demo")

    page = st.radio(
        "Navigate",
        [
            "Home",
            "Practice Studio",
            "Try the Demo",
            "Progress Tracker",
            "Research Results",
            "About & Limitations",
        ],
        index=0
    )

    st.divider()

    st.subheader("Accessibility")
    st.session_state["large_text"] = st.checkbox(
        "Large text mode",
        value=st.session_state["large_text"]
    )
    st.session_state["high_contrast"] = st.checkbox(
        "High contrast mode",
        value=st.session_state["high_contrast"]
    )

    st.caption(
        "Tip: You can use Tab, Shift+Tab, and Enter to move through most controls."
    )

apply_styles()

# -----------------------------
# Header
# -----------------------------

st.markdown(
    """
    <div class="app-hero">
        <div class="hero-title">SpeakClear AI</div>
        <div class="hero-subtitle">
            A personalized, accessible, non-clinical speech-practice app that analyzes
            short recordings, predicts clear or unclear pronunciation, and gives practice feedback.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

safe_notice()

# -----------------------------
# Pages
# -----------------------------

if page == "Home":
    st.header("Welcome")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset", "245 clips")
    with col2:
        st.metric("Best Model", "Random Forest")
    with col3:
        st.metric("Unclear F1", "0.48")
    with col4:
        st.metric("LLM Feedback", "8.2 / 10")

    st.markdown("### What the app does")

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "Practice",
            "Choose a sound, generate a practice plan, and use word banks built from the research dataset."
        )

    with c2:
        card(
            "Analyze",
            "Upload a WAV recording and receive a clear or unclear prediction with confidence."
        )

    with c3:
        card(
            "Improve",
            "Get reliable feedback, AI-rewritten feedback, and save attempts to track progress."
        )

    st.markdown("### System flow")

    st.code(
        """
Audio Upload
↓
Feature Extraction
↓
Random Forest Classifier
↓
Clear / Unclear Prediction
↓
Confidence Level
↓
Practice Priority
↓
Reliable Feedback
↓
LLM Rewrite
↓
Progress Tracking
        """,
        language="text"
    )

    st.markdown("### Target sounds")

    sound_cols = st.columns(5)

    for idx, sound in enumerate(["s", "z", "r", "th", "mixed"]):
        with sound_cols[idx]:
            st.markdown(
                f"""
                <div class="sound-card">
                    <strong>/{sound}/</strong>
                    <p>{SOUND_DESCRIPTIONS[sound]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


elif page == "Practice Studio":
    st.header("Practice Studio")

    st.write(
        "Generate a practice plan, review word banks, and see priority items from your dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        practice_sound = st.selectbox(
            "Choose a sound to practice",
            ["s", "z", "r", "th", "mixed"]
        )

    with col2:
        difficulty = st.selectbox(
            "Choose practice length",
            ["Quick practice", "Normal practice", "Long practice"]
        )

    st.markdown("### Sound description")
    st.info(SOUND_DESCRIPTIONS[practice_sound])

    st.markdown("### Word bank")

    word_cols = st.columns(4)
    for i, word in enumerate(WORD_BANK[practice_sound]):
        with word_cols[i % 4]:
            st.markdown(f'<span class="badge">{word}</span>', unsafe_allow_html=True)

    if st.button("Generate Practice Plan", type="primary"):
        challenge_words, plan = make_practice_plan(practice_sound, difficulty)
        st.session_state["challenge_words"] = challenge_words
        st.session_state["practice_plan"] = plan

    if "practice_plan" in st.session_state:
        st.markdown("### Your practice plan")

        for number, step in enumerate(st.session_state["practice_plan"], start=1):
            st.write(f"{number}. {step}")

        plan_text = "\n".join(st.session_state["practice_plan"])

        st.download_button(
            label="Download Practice Plan",
            data=plan_text,
            file_name=f"practice_plan_{practice_sound}.txt",
            mime="text/plain"
        )

    st.markdown("### Personal practice priorities")

    feedback_path = RESULTS_DIR / "practice_feedback_by_item.csv"
    feedback_df = load_csv_if_exists(feedback_path)

    if feedback_df is not None and "target_sound" in feedback_df.columns:
        filtered = feedback_df[
            feedback_df["target_sound"].astype(str).str.lower() == practice_sound
        ]

        if not filtered.empty:
            show_cols = [
                col for col in ["word", "target_sound", "unclear_rate", "priority", "feedback"]
                if col in filtered.columns
            ]
            st.dataframe(filtered[show_cols].head(10), use_container_width=True)
        else:
            st.info("No saved priority items found for this sound.")
    else:
        st.info("No practice priority file found yet.")

    st.markdown("### Recording tips")
    st.write(
        """
1. Record in a quiet room.
2. Keep the microphone distance consistent.
3. Use one word or sentence per recording.
4. Use filenames like `bath-1.wav`, `right-3.wav`, or `zero-2.wav`.
5. Upload the recording in the Try the Demo page.
        """
    )


elif page == "Try the Demo":
    st.header("Try the Demo")

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python3 train_save_model.py` first.")
        st.stop()

    model = load_model()

    st.markdown("### Step 1: Choose target sound")

    selected_sound = st.selectbox(
        "Choose the target sound",
        ["s", "z", "r", "th", "mixed"]
    )

    st.markdown("### Step 2: Upload recording")

    uploaded_file = st.file_uploader(
        "Upload a WAV audio recording",
        type=["wav"],
        help="Upload a short WAV file, ideally one word or one short sentence."
    )

    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format="audio/wav")

        item_name, inferred_sound = infer_target_sound(uploaded_file.name)
        target_sound = selected_sound

        st.markdown("### Detected recording information")

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.metric("Uploaded item", item_name)

        with info_col2:
            st.metric("Detected sound", f"/{inferred_sound}/" if inferred_sound else "Unknown")

        with info_col3:
            st.metric("Selected sound", f"/{selected_sound}/")

        if inferred_sound is not None and inferred_sound != selected_sound:
            st.warning(
                f"This file looks like a /{inferred_sound}/ recording, but you selected /{selected_sound}/. "
                f"The app will use /{inferred_sound}/ for feedback."
            )
            target_sound = inferred_sound

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with st.spinner("Analyzing audio..."):
            features = extract_features(tmp_path)
            prediction = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                classes = list(model.classes_)
                confidence = probabilities[classes.index(prediction)] * 100
            else:
                confidence = 0

        confidence_level = get_confidence_level(confidence)
        practice_priority = get_practice_priority(prediction, confidence)

        st.markdown("### Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if str(prediction).lower() == "clear":
                st.success("Prediction: Clear")
            else:
                st.error("Prediction: Unclear")

        with result_col2:
            st.metric("Confidence", f"{confidence:.1f}%")
            st.progress(int(confidence))

        with result_col3:
            st.metric("Practice Priority", practice_priority)

        st.info(f"Confidence level: {confidence_level}")

        reliable_feedback = make_reliable_feedback(
            prediction,
            confidence,
            target_sound,
            item_name
        )

        llama_feedback = make_llama_feedback(
            reliable_feedback,
            prediction
        )

        st.markdown("### Reliable practice feedback")
        st.write(reliable_feedback)

        st.markdown("### AI-rewritten feedback")
        st.write(llama_feedback)

        feedback_report = f"""
SpeakClear AI Feedback Report

File: {uploaded_file.name}
Uploaded item: {item_name}
Target sound: {target_sound}
Prediction: {prediction}
Confidence: {confidence:.1f}%
Confidence level: {confidence_level}
Practice priority: {practice_priority}

Reliable feedback:
{reliable_feedback}

AI-rewritten feedback:
{llama_feedback}

Disclaimer:
This is practice feedback only. It is not a diagnosis or treatment tool.
"""

        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                label="Download Feedback Report",
                data=feedback_report,
                file_name=f"feedback_{item_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )

        with col_b:
            if st.button("Save Attempt to Progress Tracker", type="primary"):
                save_progress_log({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_name": uploaded_file.name,
                    "item_name": item_name,
                    "target_sound": target_sound,
                    "prediction": prediction,
                    "confidence": round(confidence, 1),
                    "confidence_level": confidence_level,
                    "practice_priority": practice_priority,
                    "reliable_feedback": reliable_feedback,
                    "llm_feedback": llama_feedback,
                })

                st.success("Saved to progress tracker.")

        with st.expander("Technical details"):
            st.write(
                "This prediction is based on MFCCs, zero-crossing rate, spectral centroid, "
                "spectral rolloff, and RMS energy."
            )
            st.write(
                "The model was trained on a personal 245-clip speech-practice dataset."
            )
            st.write(
                "The LLM layer rewrites reliable rule-based feedback and falls back to the reliable version if needed."
            )


elif page == "Progress Tracker":
    st.header("Progress Tracker")

    progress_df = load_progress_log()

    if progress_df.empty:
        st.info("No saved attempts yet. Use the Try the Demo page and click Save Attempt.")
    else:
        total_attempts = len(progress_df)
        clear_count = (progress_df["prediction"].astype(str).str.lower() == "clear").sum()
        unclear_count = (progress_df["prediction"].astype(str).str.lower() == "unclear").sum()
        avg_confidence = progress_df["confidence"].mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Saved Attempts", total_attempts)

        with col2:
            st.metric("Clear", clear_count)

        with col3:
            st.metric("Unclear", unclear_count)

        with col4:
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")

        st.markdown("### Practice history")
        st.dataframe(progress_df, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### Attempts by target sound")
            st.bar_chart(progress_df["target_sound"].value_counts())

        with chart_col2:
            st.markdown("### Clear vs unclear")
            st.bar_chart(progress_df["prediction"].value_counts())

        csv_data = progress_df.to_csv(index=False)

        st.download_button(
            label="Download Progress Log",
            data=csv_data,
            file_name="practice_progress_log.csv",
            mime="text/csv"
        )


elif page == "Research Results":
    st.header("Research Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Clips", "245")
    with col2:
        st.metric("Clear Clips", "188")
    with col3:
        st.metric("Unclear Clips", "57")
    with col4:
        st.metric("Best Model", "Random Forest")

    st.markdown("### Model evaluation")

    model_results_path = RESULTS_DIR / "model_results.csv"
    model_df = load_csv_if_exists(model_results_path)

    if model_df is not None:
        st.dataframe(model_df, use_container_width=True)
    else:
        st.info("model_results.csv not found in results/.")

    st.write(
        "The Random Forest model achieved 75.8% accuracy and an unclear-class F1 score of 0.48. "
        "The Always Clear baseline achieved 77.4% accuracy but had an unclear-class F1 score of 0.00, "
        "showing that accuracy alone was misleading."
    )

    st.markdown("### Pattern analysis")

    sound_graph = RESULTS_DIR / "unclear_rate_by_sound.png"
    clip_type_graph = RESULTS_DIR / "unclear_rate_by_clip_type.png"
    top_items_graph = RESULTS_DIR / "top10_unclear_items.png"

    if sound_graph.exists():
        st.image(str(sound_graph), caption="Unclear Rate by Target Sound", use_container_width=True)

    if clip_type_graph.exists():
        st.image(str(clip_type_graph), caption="Unclear Rate by Clip Type", use_container_width=True)

    if top_items_graph.exists():
        st.image(str(top_items_graph), caption="Top 10 Hardest Words and Sentences", use_container_width=True)

    st.markdown("### LLM feedback evaluation")

    llama_summary_path = RESULTS_DIR / "llama_feedback_evaluation_summary.csv"
    llama_df = load_csv_if_exists(llama_summary_path)

    if llama_df is not None:
        st.dataframe(llama_df, use_container_width=True)
    else:
        st.info("llama_feedback_evaluation_summary.csv not found in results/.")

    st.write(
        "The LLM-assisted feedback layer achieved an average rubric score of 8.2 out of 10. "
        "Safety was the strongest category, while usefulness was the weakest category."
    )


elif page == "About & Limitations":
    st.header("About & Limitations")

    st.markdown("### Research question")
    st.write(
        "**Can machine learning analyze personalized speech-pronunciation patterns and generate useful "
        "non-clinical feedback for speech practice?**"
    )

    st.markdown("### Hypothesis")
    st.write(
        "If acoustic features from personalized speech-practice recordings are used to train a "
        "machine-learning classifier, then the system will be able to identify patterns in clear and "
        "unclear pronunciation clips and generate useful non-clinical feedback for practice."
    )

    st.markdown("### Accessibility design choices")
    st.write(
        """
- Large text mode
- High contrast mode
- Clear headings and simple language
- Keyboard-friendly Streamlit controls
- No color-only meaning: predictions also use text labels
- Downloadable reports for offline review
- Safety warning on every app session
- Short feedback instead of long paragraphs
        """
    )

    st.markdown("### Limitations")
    st.write(
        """
- The dataset contains recordings from one speaker, so results may not generalize to other speakers.
- Labels were manually assigned and may be subjective.
- The model is not clinically validated.
- The app should not be used for diagnosis or treatment.
- The LLM feedback layer can sometimes be vague or inconsistent.
- Future work could include more speakers, expert label review, and deeper audio models such as Wav2Vec2.
        """
    )

    st.markdown("### Technology used")
    st.write(
        """
- Python
- Streamlit
- librosa
- scikit-learn
- Random Forest
- Ollama / Llama
- pandas
        """
    )

st.markdown(
    """
    <div class="footer-note">
        SpeakClear AI is an independent research prototype for non-clinical speech practice.
        It should not be used as a medical, diagnostic, or therapeutic tool.
    </div>
    """,
    unsafe_allow_html=True
)
