# SpeakClear AI

SpeakClear AI is a non-clinical speech-practice prototype that uses machine learning to analyze short pronunciation recordings and generate personalized practice feedback.

## Research Question

Can machine learning analyze personalized speech-pronunciation patterns and generate useful non-clinical feedback for speech practice?

## Features

- Accessible UI with large text, high contrast, labeled inputs, and keyboard-friendly controls
- Practice Studio with sound selector, expanded word bank, difficulty filters, random challenges, and downloadable plans
- WAV upload as the main reliable analysis option
- Optional in-browser recording with `st.audio_input`
- Clear / unclear prediction from a Random Forest model
- Confidence, practice priority, and non-clinical feedback
- Rule-based feedback by default, with optional local Llama rewrite
- Progress Tracker with charts and a downloadable attempt log
- Research Results page with model tables, graphs, and plain-English explanations

## Dataset

The project uses a personal 245-clip speech-practice dataset.

- 188 clear clips
- 57 unclear clips
- Target sounds: /s/, /z/, /r/, /th/, and mixed /s/ + /z/ sentences

Raw voice recordings are not included in this public repository.

## Model Results

The best model was Random Forest.

- Accuracy: 75.8%
- Unclear precision: 46.7%
- Unclear recall: 50.0%
- Unclear F1: 0.48

The Always Clear baseline had 77.4% accuracy but 0.00 unclear F1, showing that accuracy alone was misleading.

## LLM Feedback Evaluation

The LLM-assisted feedback layer was evaluated with a 10-example rubric.

Average score: 8.2 / 10

The hosted version uses rule-based fallback feedback if local Llama / Ollama is not available.

## How to Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

This app is meant to run on Streamlit Community Cloud.

- `requirements.txt` lists Python packages
- `packages.txt` installs `libsndfile1` and `ffmpeg` for audio processing
- The model loads from `models/speakclear_random_forest.joblib`
- Research graphs stay in `results/`
- Ollama is optional. If it is not available, the app uses safe rule-based feedback

## Important Notes

- This is non-clinical practice feedback only.
- It is not a diagnosis, treatment tool, or replacement for a speech-language pathologist.
- Browser recording is optional. WAV upload is the most reliable way to analyze a clip.
