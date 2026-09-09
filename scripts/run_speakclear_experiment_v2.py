from pathlib import Path
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

INPUT_CSV = Path("data/speakclear_master.csv")
RESULTS_DIR = Path("results_v2")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    for feat in [zcr, centroid, rolloff, rms]:
        features.append(float(np.mean(feat)))
        features.append(float(np.std(feat)))

    return np.array(features, dtype=np.float32)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    labels = ["clear", "unclear"]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        preds,
        labels=labels,
        zero_division=0,
    )

    row = {
        "dataset": "SpeakClear",
        "split_type": "single_speaker_stratified",
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
        "clear_precision": precision[0],
        "clear_recall": recall[0],
        "clear_f1": f1[0],
        "unclear_precision": precision[1],
        "unclear_recall": recall[1],
        "unclear_f1": f1[1],
        "test_clear_support": support[0],
        "test_unclear_support": support[1],
    }

    cm = confusion_matrix(y_test, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(RESULTS_DIR / f"{filename}_confusion_matrix.png", dpi=200)
    plt.close()

    return row, preds


def main():
    df = pd.read_csv(INPUT_CSV)

    print("Rows:", len(df))
    print("\nBy label:")
    print(df["label"].value_counts())
    print("\nBy sound and label:")
    print(pd.crosstab(df["target_sound"], df["label"]))
    print("\nSpeakers:", df["speaker_id"].nunique())

    X = []
    y = []

    for _, row in df.iterrows():
        X.append(extract_features(row["audio_path"]))
        y.append(row["label"])

    X = np.vstack(X)
    y = np.array(y)

    print("\nFeature matrix:", X.shape)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_df = df.iloc[test_idx].copy()

    print("\nSplit type: single_speaker_stratified")
    print("Train labels:", pd.Series(y_train).value_counts().to_dict())
    print("Test labels:", pd.Series(y_test).value_counts().to_dict())

    models = {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    }

    results = []
    all_predictions = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        row, preds = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(row)
        print(row)

        pred_df = test_df.copy()
        pred_df["model"] = name
        pred_df["prediction"] = preds
        all_predictions.append(pred_df)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "speakclear_model_results_v2.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(RESULTS_DIR / "speakclear_test_predictions_v2.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / "speakclear_model_results_v2.csv")
    print(RESULTS_DIR / "speakclear_test_predictions_v2.csv")
    print(RESULTS_DIR / "*_confusion_matrix.png")


if __name__ == "__main__":
    main()
