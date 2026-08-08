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
    layout="wide"
)

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
    elif confidence >= 70:
        return "Medium confidence"
    return "Low confidence — review manually"


def get_practice_priority(prediction, confidence):
    prediction = prediction.lower()

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
    if target_sound == "mixed":
        sound_text = "/s/ and /z/ sounds"
    else:
        sound_text = f"/{target_sound}/ sound"

    if item_name:
        item_text = f"for '{item_name}' "
    else:
        item_text = ""

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

            if prediction == "clear":
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

    return selected, plan


st.title("🎙️ SpeakClear AI")
st.caption("Machine Learning for Personalized Speech-Practice Feedback")

home_tab, practice_tab, demo_tab, progress_tab, results_tab, about_tab = st.tabs(
    [
        "Home",
        "Practice Mode",
        "Try the Demo",
        "Progress Tracker",
        "Research Results",
        "About & Limitations",
    ]
)

with home_tab:
    st.header("Personalized AI Speech-Practice Feedback")

    st.write(
        "SpeakClear AI is a non-clinical prototype that analyzes short speech-practice "
        "recordings and gives personalized feedback based on pronunciation patterns."
    )

    st.warning(
        "This app is for speech practice only. It is not a diagnosis, treatment tool, "
        "or replacement for a speech-language pathologist."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Size", "245 clips")

    with col2:
        st.metric("Best Model", "Random Forest")

    with col3:
        st.metric("Unclear F1", "0.48")

    with col4:
        st.metric("LLM Feedback", "8.2 / 10")

    st.subheader("How the System Works")

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
Confidence + Practice Priority
↓
Reliable Rule-Based Feedback
↓
LLM-Assisted Feedback Rewrite
↓
Progress Tracking
        """,
        language="text"
    )

    st.subheader("What makes this personalized?")

    st.write(
        "Instead of using only general pronunciation exercises, SpeakClear AI uses a personal "
        "speech-practice dataset to identify which sounds and words were most difficult. "
        "The system then turns those patterns into practice priorities and feedback."
    )


with practice_tab:
    st.header("Practice Mode")

    st.write(
        "Use this page to generate a practice plan before uploading recordings to the demo."
    )

    col1, col2 = st.columns(2)

    with col1:
        practice_sound = st.selectbox(
            "Choose a sound to practice:",
            ["s", "z", "r", "th", "mixed"],
            key="practice_sound"
        )

    with col2:
        difficulty = st.selectbox(
            "Choose practice length:",
            ["Quick practice", "Normal practice", "Long practice"]
        )

    st.subheader("Word Bank")

    selected_words = WORD_BANK[practice_sound]
    st.write(", ".join(selected_words))

    if st.button("Generate Practice Plan"):
        challenge_words, plan = make_practice_plan(practice_sound, difficulty)

        st.session_state["challenge_words"] = challenge_words
        st.session_state["practice_plan"] = plan

    if "practice_plan" in st.session_state:
        st.subheader("Your Practice Plan")

        for step in st.session_state["practice_plan"]:
            st.write(f"- {step}")

        plan_text = "\n".join(st.session_state["practice_plan"])

        st.download_button(
            label="Download Practice Plan",
            data=plan_text,
            file_name=f"practice_plan_{practice_sound}.txt",
            mime="text/plain"
        )

    st.subheader("Recording Tips")

    st.write(
        """
1. Record in a quiet room.
2. Keep your phone or microphone the same distance away each time.
3. Say one word or sentence per recording.
4. Use filenames like `bath-1.wav`, `bath-2.wav`, or `right-3.wav`.
5. Upload the recording in the Try the Demo tab.
        """
    )

    st.subheader("Personal Practice Priorities")

    feedback_path = RESULTS_DIR / "practice_feedback_by_item.csv"

    if feedback_path.exists():
        feedback_df = pd.read_csv(feedback_path)

        if "target_sound" in feedback_df.columns:
            filtered = feedback_df[feedback_df["target_sound"].astype(str).str.lower() == practice_sound]

            if not filtered.empty:
                st.write("Hardest items from your dataset for this sound:")
                st.dataframe(
                    filtered[["word", "target_sound", "unclear_rate", "priority"]].head(10),
                    use_container_width=True
                )
            else:
                st.info("No saved priority items found for this sound.")
    else:
        st.info("practice_feedback_by_item.csv not found yet.")


with demo_tab:
    st.header("Try the Speech-Practice Demo")

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python3 train_save_model.py` first.")
    else:
        model = load_model()

        selected_sound = st.selectbox(
            "Choose the target sound:",
            ["s", "z", "r", "th", "mixed"],
            key="demo_sound"
        )

        uploaded_file = st.file_uploader(
            "Upload a WAV audio recording",
            type=["wav"]
        )

        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes, format="audio/wav")

            item_name, inferred_sound = infer_target_sound(uploaded_file.name)
            target_sound = selected_sound

            st.markdown("### Detected Recording Info")

            info_col1, info_col2, info_col3 = st.columns(3)

            with info_col1:
                st.write(f"**Uploaded item:** `{item_name}`")

            with info_col2:
                if inferred_sound:
                    st.write(f"**Detected target sound:** `/{inferred_sound}/`")
                else:
                    st.write("**Detected target sound:** Unknown")

            with info_col3:
                st.write(f"**Selected sound:** `/{selected_sound}/`")

            if inferred_sound is not None and inferred_sound != selected_sound:
                st.warning(
                    f"This file looks like a /{inferred_sound}/ recording, but you selected /{selected_sound}/. "
                    f"Using /{inferred_sound}/ for feedback."
                )
                target_sound = inferred_sound

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

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

            st.markdown("## Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                if prediction == "clear":
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

            st.markdown("## Reliable Practice Feedback")
            st.write(reliable_feedback)

            st.markdown("## AI-Rewritten Feedback")
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

            st.download_button(
                label="Download This Feedback",
                data=feedback_report,
                file_name=f"feedback_{item_name.replace(' ', '_')}.txt",
                mime="text/plain"
            )

            if st.button("Save This Attempt to Progress Tracker"):
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

            with st.expander("Show technical model details"):
                st.write(
                    "This prediction is based on acoustic features such as MFCCs, "
                    "zero-crossing rate, spectral centroid, spectral rolloff, and RMS energy."
                )
                st.write("The model was trained on a personal 245-clip speech-practice dataset.")
                st.write("The AI feedback layer rewrites reliable rule-based feedback, with backup safety rules.")


with progress_tab:
    st.header("Progress Tracker")

    st.write(
        "This page shows saved demo attempts. Use the button in the Try the Demo tab to save predictions here."
    )

    progress_df = load_progress_log()

    if progress_df.empty:
        st.info("No saved attempts yet.")
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
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

        st.subheader("Prediction History")
        st.dataframe(progress_df, use_container_width=True)

        st.subheader("Attempts by Target Sound")

        sound_counts = progress_df["target_sound"].value_counts()
        st.bar_chart(sound_counts)

        st.subheader("Clear vs Unclear")

        prediction_counts = progress_df["prediction"].value_counts()
        st.bar_chart(prediction_counts)

        csv_data = progress_df.to_csv(index=False)

        st.download_button(
            label="Download Progress Log",
            data=csv_data,
            file_name="practice_progress_log.csv",
            mime="text/csv"
        )


with results_tab:
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

    st.subheader("Model Evaluation")

    model_results_path = RESULTS_DIR / "model_results.csv"

    if model_results_path.exists():
        model_df = pd.read_csv(model_results_path)
        st.dataframe(model_df, use_container_width=True)
    else:
        st.info("model_results.csv not found in results/.")

    st.write(
        "The Random Forest model achieved 75.8% accuracy and an unclear-class F1 score of 0.48. "
        "The Always Clear baseline achieved 77.4% accuracy but had an unclear-class F1 score of 0.00, "
        "showing that accuracy alone was misleading for this imbalanced dataset."
    )

    st.subheader("Pronunciation Pattern Analysis")

    sound_graph = RESULTS_DIR / "unclear_rate_by_sound.png"
    clip_type_graph = RESULTS_DIR / "unclear_rate_by_clip_type.png"
    top_items_graph = RESULTS_DIR / "top10_unclear_items.png"

    if sound_graph.exists():
        st.image(str(sound_graph), caption="Unclear Rate by Target Sound", use_container_width=True)

    if clip_type_graph.exists():
        st.image(str(clip_type_graph), caption="Unclear Rate by Clip Type", use_container_width=True)

    if top_items_graph.exists():
        st.image(str(top_items_graph), caption="Top 10 Hardest Words / Sentences", use_container_width=True)

    st.subheader("LLM Feedback Evaluation")

    llama_summary_path = RESULTS_DIR / "llama_feedback_evaluation_summary.csv"

    if llama_summary_path.exists():
        llama_df = pd.read_csv(llama_summary_path)
        st.dataframe(llama_df, use_container_width=True)
    else:
        st.info("llama_feedback_evaluation_summary.csv not found in results/.")

    st.write(
        "The LLM-assisted feedback layer achieved an average rubric score of 8.2 out of 10. "
        "The strongest category was safety, while the weakest category was usefulness. "
        "This suggests that the LLM feedback was usually safe and understandable, but sometimes too general "
        "or slightly inconsistent with the classifier output."
    )


with about_tab:
    st.header("About SpeakClear AI")

    st.subheader("Research Question")

    st.write(
        "**Can machine learning analyze personalized speech-pronunciation patterns and generate useful "
        "non-clinical feedback for speech practice?**"
    )

    st.subheader("Hypothesis")

    st.write(
        "If acoustic features from personalized speech-practice recordings are used to train a "
        "machine-learning classifier, then the system will be able to identify patterns in clear and "
        "unclear pronunciation clips and generate useful non-clinical feedback for practice."
    )

    st.subheader("What this app does")

    st.write(
        """
- Analyzes uploaded WAV recordings
- Predicts clear or unclear pronunciation
- Shows model confidence
- Assigns a practice priority
- Gives reliable practice feedback
- Rewrites feedback with a local LLM
- Tracks saved practice attempts
- Displays research results and graphs
        """
    )

    st.subheader("Limitations")

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

    st.subheader("Technology Used")

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
