from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv("metadata.csv")
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["target_sound"] = df["target_sound"].astype(str).str.strip().str.lower()
df["word"] = df["word"].astype(str).str.strip()

df = df[df["label"].isin(["clear", "unclear"])].copy()
df["clip_type"] = df["word"].apply(lambda x: "sentence" if " " in x else "word")

summary = (
    df.groupby(["word", "target_sound", "clip_type"])
    .agg(
        total=("label", "count"),
        unclear_count=("label", lambda x: (x == "unclear").sum())
    )
    .reset_index()
)

summary["unclear_rate"] = summary["unclear_count"] / summary["total"]
summary = summary.sort_values(["unclear_rate", "unclear_count"], ascending=False)

def make_feedback(row):
    word = row["word"]
    sound = row["target_sound"]
    rate = row["unclear_rate"]
    clip_type = row["clip_type"]

    target_phrase = "the /s/ and /z/ sounds" if sound == "mixed" else f"the /{sound}/ sound"

    if rate >= 0.6:
        level = "high-priority"
        advice = (
            f"Practice '{word}' slowly and repeat it in short sets. "
            f"Focus on keeping {target_phrase} steady and controlled. "
            f"Because this item had a high unclear rate, it may be useful to practice it before moving to faster speech."
        )
    elif rate >= 0.3:
        level = "medium-priority"
        advice = (
            f"'{word}' showed occasional unclear pronunciation. "
            f"Practice it at a normal pace, then slowly increase speed while keeping {target_phrase} consistent."
        )
    else:
        level = "low-priority"
        advice = (
            f"'{word}' was mostly clear. Continue practicing it for consistency, but prioritize harder words or sentences first."
        )

    if clip_type == "sentence":
        advice += " Since this is a sentence, try breaking it into smaller parts before saying the full phrase."

    advice += " This is non-clinical practice feedback, not a diagnosis."

    return level, advice

summary[["priority", "feedback"]] = summary.apply(
    lambda row: pd.Series(make_feedback(row)),
    axis=1
)

summary.to_csv(RESULTS_DIR / "practice_feedback_by_item.csv", index=False)

with open(RESULTS_DIR / "practice_feedback_report.txt", "w") as f:
    f.write("SpeakClear AI Practice Feedback Report\n")
    f.write("=====================================\n\n")

    f.write("Dataset Summary\n")
    f.write(f"Total clips: {len(df)}\n")
    f.write(f"Clear clips: {(df['label'] == 'clear').sum()}\n")
    f.write(f"Unclear clips: {(df['label'] == 'unclear').sum()}\n\n")

    f.write("Top Practice Priorities\n")
    f.write("-----------------------\n")
    for _, row in summary.head(10).iterrows():
        f.write(f"\nItem: {row['word']}\n")
        f.write(f"Target sound: {row['target_sound']}\n")
        f.write(f"Clip type: {row['clip_type']}\n")
        f.write(f"Unclear rate: {row['unclear_rate']:.2f}\n")
        f.write(f"Priority: {row['priority']}\n")
        f.write(f"Feedback: {row['feedback']}\n")

print("Created:")
print("results/practice_feedback_by_item.csv")
print("results/practice_feedback_report.txt")
print("\nTop feedback items:")
print(summary[["word", "target_sound", "unclear_rate", "priority", "feedback"]].head(10))
