from pathlib import Path
import tempfile
import requests

import streamlit as st
import librosa
import numpy as np
import joblib

MODEL_PATH = Path("models/speakclear_random_forest.joblib")

st.set_page_config(
    page_title="SpeakClear AI",
    page_icon="🎙️",
    layout="centered"
)

R_WORDS = {
    "red", "race", "right", "river", "around",
    "carrot", "mirror", "story", "car", "far"
}

TH_WORDS = {
    "think", "thin", "three", "thirty", "thank",
    "birthday", "nothing", "healthy", "bath", "teeth"
}

S_WORDS = {
    "sun", "seal", "seven", "sister", "science",
    "soup", "same", "bus", "yes", "class"
}

Z_WORDS = {
    "zoo", "zero", "zebra", "zipper", "zone",
    "buzz", "lazy", "puzzle", "music", "cheese"
}


def infer_target_sound(filename):
    stem = Path(filename).stem.lower()

    if "-" in stem:
        word = stem.rsplit("-", 1)[0]
    else:
        word = stem

    word = word.replace("_", " ").strip()
    first_word = word.split()[0] if word.split() else word

    if word in R_WORDS or first_word in R_WORDS:
        return "r"

    if word in TH_WORDS or first_word in TH_WORDS:
        return "th"

    if word in S_WORDS or first_word in S_WORDS:
        return "s"

    if word in Z_WORDS or first_word in Z_WORDS:
        return "z"

    if "zach said science was easy" in word:
        return "mixed"

    if "susan" in word or "students" in word or "science" in word:
        return "mixed"

    if "zebra" in word or "puzzle" in word or "cheese" in word:
        return "z"

    return None


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


def make_feedback(prediction, confidence, target_sound):
    if target_sound == "mixed":
        sound_text = "/s/ and /z/ sounds"
    else:
        sound_text = f"/{target_sound}/ sound"

    if prediction == "clear":
        return (
            f"The recording was predicted as clear with {confidence:.1f}% confidence. "
            f"Keep practicing the {sound_text} slowly and consistently. "
            f"This is practice feedback only."
        )

    return (
        f"The recording was predicted as unclear with {confidence:.1f}% confidence. "
        f"Practice the word slowly and repeat it in short sets while focusing on the {sound_text}. "
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


def make_llama_feedback(basic_feedback):
    prompt = f"""
You are writing user-facing feedback for a speech-practice website.

Rewrite the original feedback into exactly 2 short sentences.

Rules:
- Only output the final feedback.
- Do not introduce the response.
- Do not say "Here are two sentences."
- Do not say "rewriting the original feedback."
- Do not add new information.
- Do not mention mistakes unless the original feedback says unclear.
- Do not mention therapy, treatment, diagnosis, disorder, or speech-language pathologist.
- Do not use technical speech terms like alveolar ridge, guttural, placement, or articulation.
- Keep it simple, supportive, and non-clinical.
- The final sentence must include: "This is practice feedback only."

Original feedback:
{basic_feedback}
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

            if llama_text and not any(phrase in llama_text.lower() for phrase in banned_phrases):
                return llama_text

        return basic_feedback

    except Exception:
        return basic_feedback


st.title("🎙️ SpeakClear AI")
st.subheader("Personalized Speech-Practice Feedback")

st.write(
    "Upload a short speech recording. The model will analyze the audio and predict "
    "whether the pronunciation sounds clear or unclear."
)

st.warning(
    "SpeakClear AI is a non-clinical prototype. It is not a diagnosis, treatment tool, "
    "or replacement for a speech-language pathologist."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python3 train_save_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

selected_sound = st.selectbox(
    "Choose the target sound:",
    ["s", "z", "r", "th", "mixed"]
)

uploaded_file = st.file_uploader(
    "Upload a WAV audio recording",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    inferred_sound = infer_target_sound(uploaded_file.name)
    target_sound = selected_sound

    if inferred_sound is not None and inferred_sound != selected_sound:
        st.warning(
            f"This file looks like a /{inferred_sound}/ recording, but you selected /{selected_sound}/. "
            f"Using /{inferred_sound}/ for feedback."
        )
        target_sound = inferred_sound

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    features = extract_features(tmp_path)
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        confidence = probabilities[classes.index(prediction)] * 100
    else:
        confidence = 0

    st.markdown("## Result")

    if prediction == "clear":
        st.success(f"Prediction: Clear ({confidence:.1f}% confidence)")
    else:
        st.error(f"Prediction: Unclear ({confidence:.1f}% confidence)")

    st.markdown("## AI Practice Feedback")

    basic_feedback = make_feedback(prediction, confidence, target_sound)
    llama_feedback = make_llama_feedback(basic_feedback)

    st.write(llama_feedback)

    with st.expander("Show rule-based backup feedback"):
        st.write(basic_feedback)

    st.markdown("## Research Note")
    st.write(
        "This prediction is based on acoustic features such as MFCCs, zero-crossing rate, "
        "spectral centroid, spectral rolloff, and RMS energy. The model was trained on a "
        "personal 245-clip speech-practice dataset. The feedback is generated using an "
        "LLM-assisted feedback layer with a rule-based backup."
    )

else:
    st.info("Upload a WAV file to get a prediction and practice feedback.")
