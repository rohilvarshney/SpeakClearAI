from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

AUDIO_DIR = Path("wav")
METADATA_PATH = Path("metadata.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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

    return np.array(features)

df = pd.read_csv(METADATA_PATH)
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["clear", "unclear"])].copy()

X = []
y = []

for _, row in df.iterrows():
    audio_path = AUDIO_DIR / row["file_name"]
    if audio_path.exists():
        X.append(extract_features(audio_path))
        y.append(row["label"])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print(classification_report(y_test, preds, zero_division=0))

joblib.dump(model, MODEL_DIR / "speakclear_random_forest.joblib")

print("Saved model to models/speakclear_random_forest.joblib")
