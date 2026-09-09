from pathlib import Path
import pandas as pd

INPUT_CSV = Path("metadata.csv")
OUTPUT_CSV = Path("data/speakclear_master.csv")
WAV_DIR = Path("wav")

df = pd.read_csv(INPUT_CSV)

required = ["file_name", "word", "target_sound", "label", "speaker_id", "round"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"metadata.csv is missing columns: {missing}")

df = df.copy()
df["source"] = "speakclear"
df["audio_path"] = df["file_name"].apply(lambda x: str(WAV_DIR / str(x)))
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["target_sound"] = df["target_sound"].astype(str).str.strip().str.lower()
df["speaker_id"] = df["speaker_id"].astype(str).str.strip()

df = df[df["label"].isin(["clear", "unclear"])]
df = df[df["target_sound"].isin(["s", "z", "r", "th", "mixed"])]

df["audio_exists"] = df["audio_path"].apply(lambda p: Path(p).exists())

missing_audio = df[~df["audio_exists"]]
if len(missing_audio) > 0:
    print("WARNING: Some audio files are missing:")
    print(missing_audio[["file_name", "audio_path"]].head(20))

df = df[df["audio_exists"]].copy()

final_cols = [
    "source",
    "file_name",
    "audio_path",
    "speaker_id",
    "word",
    "target_sound",
    "label",
    "round",
]

df[final_cols].to_csv(OUTPUT_CSV, index=False)

print("Created:", OUTPUT_CSV)
print("Total rows:", len(df))
print("\nBy label:")
print(df["label"].value_counts())
print("\nBy target sound:")
print(df["target_sound"].value_counts())
print("\nBy sound and label:")
print(pd.crosstab(df["target_sound"], df["label"]))
print("\nSpeakers:", df["speaker_id"].nunique())
