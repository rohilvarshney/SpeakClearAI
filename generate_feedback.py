import pandas as pd
from pathlib import Path

METADATA_PATH = Path("metadata.csv")
OUTPUT_PATH = Path("results/feedback_examples.csv")

def generate_feedback(word, target_sound, label):
    if label == "clear":
        return (
            f"Your pronunciation of '{word}' sounded clear for the /{target_sound}/ sound. "
            f"Keep practicing it slowly and try to make each repetition consistent."
        )
    else:
        return (
            f"Your pronunciation of '{word}' may need more practice for the /{target_sound}/ sound. "
            f"Try saying the word slowly, focusing on steady airflow and a clean /{target_sound}/ sound. "
            f"This is non-clinical practice feedback, not a diagnosis."
        )

df = pd.read_csv(METADATA_PATH)
df["label"] = df["label"].astype(str).str.strip().str.lower()

df["feedback"] = df.apply(
    lambda row: generate_feedback(row["word"], row["target_sound"], row["label"]),
    axis=1
)

OUTPUT_PATH.parent.mkdir(exist_ok=True)
df[["file_name", "word", "target_sound", "label", "feedback"]].to_csv(OUTPUT_PATH, index=False)

print(f"Saved feedback examples to {OUTPUT_PATH}")
print(df[["word", "label", "feedback"]].head(10))

