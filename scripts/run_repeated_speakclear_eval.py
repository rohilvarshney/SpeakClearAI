from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

INPUT_CSV = Path("data/speakclear_master.csv")
OUT_DIR = Path("results_final")
OUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 30
TEST_SIZE = 0.25
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


def get_models(seed):
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        ),
    }


def evaluate(y_test, preds):
    labels = ["clear", "unclear"]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        preds,
        labels=labels,
        zero_division=0,
    )

    return {
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


def main():
    df = pd.read_csv(INPUT_CSV)

    print("Rows:", len(df))
    print("Labels:")
    print(df["label"].value_counts())

    X = np.vstack([extract_features(p) for p in df["audio_path"]])
    y = df["label"].to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    rows = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, model in get_models(RANDOM_STATE + split_id).items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            row = evaluate(y_test, preds)
            row["split_id"] = split_id
            row["model"] = model_name
            rows.append(row)

        print(f"Finished split {split_id}/{N_SPLITS}")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "speakclear_repeated_eval_raw.csv", index=False)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "unclear_precision",
        "unclear_recall",
        "unclear_f1",
    ]

    summary = raw.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    summary = summary.sort_values("unclear_f1_mean", ascending=False)
    summary.to_csv(OUT_DIR / "speakclear_repeated_eval_summary.csv", index=False)

    print("\nRepeated evaluation summary:")
    print(summary)


if __name__ == "__main__":
    main()
