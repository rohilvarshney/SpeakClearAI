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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

SPEAKCLEAR_CSV = Path("data/speakclear_master.csv")
SPEECHOCEAN_CSV = Path("data/speechocean_targets.csv")
RESULTS_DIR = Path("results_transfer")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
MAX_ROWS_PER_SOUND_LABEL = 400


def extract_features_from_array(y, sr):
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != 16000:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
        sr = 16000

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


def extract_features_from_file(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return extract_features_from_array(y, sr)


def load_audio_from_hf_example(audio_info):
    audio_bytes = audio_info.get("bytes")
    audio_path = audio_info.get("path")

    if audio_bytes is not None:
        y, sr = sf.read(BytesIO(audio_bytes), dtype="float32")
    elif audio_path:
        y, sr = sf.read(audio_path, dtype="float32")
    else:
        raise ValueError("No audio bytes or path found")

    return y, sr


def balance_speechocean(df):
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


def build_speakclear_features(df):
    X = []
    y = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SpeakClear features"):
        X.append(extract_features_from_file(row["audio_path"]))
        y.append(row["label"])

    return np.vstack(X), np.array(y)


def build_speechocean_features(df):
    print("Loading SpeechOcean without torchcodec decoding...")
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
    used_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="SpeechOcean features"):
        try:
            split_name = str(row["split"])
            idx = int(row["original_index"])
            ex = split_map[split_name][idx]

            audio_y, sr = load_audio_from_hf_example(ex["audio"])
            feats = extract_features_from_array(audio_y, sr)

            X.append(feats)
            y.append(row["label"])
            used_rows.append(row.to_dict())
        except Exception:
            continue

    used_df = pd.DataFrame(used_rows)
    used_df.to_csv(RESULTS_DIR / "speechocean_transfer_used_rows.csv", index=False)

    return np.vstack(X), np.array(y)


def get_models():
    return {
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


def evaluate_model(experiment_name, model_name, model, X_train, y_train, X_test, y_test):
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
        "experiment": experiment_name,
        "model": model_name,
        "train_size": len(y_train),
        "test_size": len(y_test),
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
    plt.title(f"{experiment_name} - {model_name}")
    plt.tight_layout()

    safe_experiment = experiment_name.lower().replace(" ", "_").replace("+", "plus")
    safe_model = model_name.lower().replace(" ", "_")
    plt.savefig(RESULTS_DIR / f"{safe_experiment}_{safe_model}_confusion_matrix.png", dpi=200)
    plt.close()

    return row, preds


def main():
    speakclear_df = pd.read_csv(SPEAKCLEAR_CSV)
    speechocean_df = pd.read_csv(SPEECHOCEAN_CSV)

    speechocean_df = speechocean_df[speechocean_df["label"].isin(["clear", "unclear"])].copy()
    speechocean_df = balance_speechocean(speechocean_df)

    print("SpeakClear rows:", len(speakclear_df))
    print("SpeechOcean balanced rows:", len(speechocean_df))

    print("\nSpeakClear labels:")
    print(speakclear_df["label"].value_counts())

    print("\nSpeechOcean labels:")
    print(speechocean_df["label"].value_counts())

    X_sc, y_sc = build_speakclear_features(speakclear_df)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    sc_train_idx, sc_test_idx = next(splitter.split(X_sc, y_sc))

    X_sc_train = X_sc[sc_train_idx]
    y_sc_train = y_sc[sc_train_idx]
    X_sc_test = X_sc[sc_test_idx]
    y_sc_test = y_sc[sc_test_idx]

    speakclear_test_df = speakclear_df.iloc[sc_test_idx].copy()
    speakclear_test_df.to_csv(RESULTS_DIR / "speakclear_transfer_test_split.csv", index=False)

    print("\nSpeakClear train labels:", pd.Series(y_sc_train).value_counts().to_dict())
    print("SpeakClear test labels:", pd.Series(y_sc_test).value_counts().to_dict())

    X_so, y_so = build_speechocean_features(speechocean_df)

    print("\nSpeechOcean feature matrix:", X_so.shape)
    print("SpeechOcean labels:", pd.Series(y_so).value_counts().to_dict())

    experiments = {
        "SpeakClear Only": (X_sc_train, y_sc_train),
        "SpeechOcean Only": (X_so, y_so),
        "SpeechOcean + SpeakClear": (
            np.vstack([X_so, X_sc_train]),
            np.concatenate([y_so, y_sc_train]),
        ),
    }

    results = []
    predictions = []

    for experiment_name, (X_train, y_train) in experiments.items():
        print("\n" + "=" * 80)
        print("Experiment:", experiment_name)
        print("Training size:", len(y_train))
        print("Training labels:", pd.Series(y_train).value_counts().to_dict())

        for model_name, model in get_models().items():
            print(f"Training {model_name}...")
            row, preds = evaluate_model(
                experiment_name,
                model_name,
                model,
                X_train,
                y_train,
                X_sc_test,
                y_sc_test,
            )
            results.append(row)
            print(row)

            pred_df = speakclear_test_df.copy()
            pred_df["experiment"] = experiment_name
            pred_df["model"] = model_name
            pred_df["prediction"] = preds
            predictions.append(pred_df)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "transfer_model_results.csv", index=False)

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv(RESULTS_DIR / "transfer_test_predictions.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / "transfer_model_results.csv")
    print(RESULTS_DIR / "transfer_test_predictions.csv")
    print(RESULTS_DIR / "*_confusion_matrix.png")


if __name__ == "__main__":
    main()
