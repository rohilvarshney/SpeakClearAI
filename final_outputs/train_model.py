import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

PROJECT_DIR = Path(".")
AUDIO_DIR = PROJECT_DIR / "wav"
METADATA_PATH = PROJECT_DIR / "metadata.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=30)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Other helpful audio features
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    features = []

    # Mean and std of MFCCs
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Mean and std of other features
    for feat in [zcr, centroid, rolloff, rms]:
        features.append(np.mean(feat))
        features.append(np.std(feat))

    return np.array(features)


def main():
    df = pd.read_csv(METADATA_PATH)

    # Clean labels from Numbers export
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Keep only valid rows
    df = df[df["label"].isin(["clear", "unclear"])].copy()

    print("Dataset label counts:")
    print(df["label"].value_counts())
    print()

    X = []
    y = []

    for _, row in df.iterrows():
        audio_path = AUDIO_DIR / row["file_name"]

        if not audio_path.exists():
            print(f"Missing file: {audio_path}")
            continue

        features = extract_features(audio_path)
        X.append(features)
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "Always Clear Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    results = []

    for name, model in models.items():
        print("=" * 50)
        print(name)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label="unclear", zero_division=0)
        recall = recall_score(y_test, preds, pos_label="unclear", zero_division=0)
        f1 = f1_score(y_test, preds, pos_label="unclear", zero_division=0)

        print(classification_report(y_test, preds, zero_division=0))

        results.append({
            "model": name,
            "accuracy": acc,
            "precision_unclear": precision,
            "recall_unclear": recall,
            "f1_unclear": f1,
        })

        cm = confusion_matrix(y_test, preds, labels=["clear", "unclear"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["clear", "unclear"]
        )

        disp.plot()
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(RESULTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    print("\nFinal results:")
    print(results_df)


if __name__ == "__main__":
    main()

