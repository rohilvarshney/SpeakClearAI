from pathlib import Path
from io import BytesIO
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

TARGET_CSV = Path("data/speechocean_targets.csv")
RESULTS_DIR = Path("results_public")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
MAX_ROWS_PER_SOUND_LABEL = 400


def load_audio_from_hf_example(audio_info):
    """
    Works without torchcodec by reading raw bytes or cached path.
    """
    if audio_info is None:
        raise ValueError("Missing audio")

    audio_bytes = audio_info.get("bytes")
    audio_path = audio_info.get("path")

    if audio_bytes is not None:
        y, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
    elif audio_path:
        y, sr = sf.read(audio_path, dtype="float32")
    else:
        raise ValueError("No audio bytes or path found")

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != 16000:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
        sr = 16000

    return y, sr


def extract_features(y, sr):
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


def balance_subset(df):
    parts = []
    for sound in sorted(df["target_sound"].unique()):
        for label in sorted(df["label"].unique()):
            sub = df[(df["target_sound"] == sound) & (df["label"] == label)]
            if len(sub) == 0:
                continue
            parts.append(
                sub.sample(
                    n=min(len(sub), MAX_ROWS_PER_SOUND_LABEL),
                    random_state=RANDOM_STATE,
                )
            )

    return pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_features(df):
    print("Loading SpeechOcean train/test splits without torchcodec audio decoding...")
    train_ds = load_dataset("mispeech/speechocean762", split="train")
    test_ds = load_dataset("mispeech/speechocean762", split="test")

    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    test_ds = test_ds.cast_column("audio", Audio(decode=False))

    split_map = {
        "train": train_ds,
        "test": test_ds,
    }

    X = []
    y = []
    groups = []
    used_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        split_name = str(row["split"])
        idx = int(row["original_index"])

        try:
            ex = split_map[split_name][idx]
            audio_info = ex["audio"]
            audio_y, sr = load_audio_from_hf_example(audio_info)
            feats = extract_features(audio_y, sr)

            X.append(feats)
            y.append(row["label"])
            groups.append(row["speaker_id"])
            used_rows.append(row.to_dict())
        except Exception as e:
            continue

    return np.vstack(X), np.array(y), np.array(groups), pd.DataFrame(used_rows)


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

    return row


def main():
    df = pd.read_csv(TARGET_CSV)
    df = df[df["label"].isin(["clear", "unclear"])].copy()

    print("Original target rows:", len(df))
    print(pd.crosstab(df["target_sound"], df["label"]))

    df = balance_subset(df)

    print("\nBalanced/subsampled rows:", len(df))
    print(pd.crosstab(df["target_sound"], df["label"]))

    X, y, groups, used_df = build_features(df)

    print("\nFeature matrix:", X.shape)
    print("Labels:", pd.Series(y).value_counts().to_dict())
    print("Speakers:", len(set(groups)))

    used_df.to_csv(RESULTS_DIR / "speechocean_used_rows.csv", index=False)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\nTrain labels:", pd.Series(y_train).value_counts().to_dict())
    print("Test labels:", pd.Series(y_test).value_counts().to_dict())

    models = {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        row = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(row)
        print(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "speechocean_model_results.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / "speechocean_model_results.csv")
    print(RESULTS_DIR / "*_confusion_matrix.png")


if __name__ == "__main__":
    main()
