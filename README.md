# SpeakClear AI

SpeakClear AI is a non-clinical speech-practice prototype that uses machine learning to analyze short pronunciation recordings and generate personalized practice feedback.

## Research Question

Can machine learning analyze personalized speech-pronunciation patterns and generate useful non-clinical feedback for speech practice?

## Features

- Upload a WAV speech-practice recording
- Predict clear or unclear pronunciation
- Show model confidence
- Assign a practice priority
- Generate reliable non-clinical feedback
- Display research results and graphs
- Track saved practice attempts locally
- Accessible design with large text and high contrast modes

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

The hosted version may use rule-based fallback feedback if local Llama/Ollama is not available.

## How to Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
