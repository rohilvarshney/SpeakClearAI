from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv("metadata.csv")
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["target_sound"] = df["target_sound"].astype(str).str.strip().str.lower()
df["word"] = df["word"].astype(str).str.strip()

df = df[df["label"].isin(["clear", "unclear"])].copy()

# Add clip type: word or sentence
df["clip_type"] = df["word"].apply(lambda x: "sentence" if " " in x else "word")

# Overall counts
overall = df["label"].value_counts().reset_index()
overall.columns = ["label", "count"]
overall.to_csv(RESULTS_DIR / "overall_label_counts.csv", index=False)

# Unclear rate by target sound
sound_summary = (
    df.groupby("target_sound")
    .agg(
        total=("label", "count"),
        unclear_count=("label", lambda x: (x == "unclear").sum())
    )
    .reset_index()
)
sound_summary["unclear_rate"] = sound_summary["unclear_count"] / sound_summary["total"]
sound_summary.to_csv(RESULTS_DIR / "unclear_rate_by_sound.csv", index=False)

# Unclear rate by word/sentence
item_summary = (
    df.groupby(["word", "target_sound", "clip_type"])
    .agg(
        total=("label", "count"),
        unclear_count=("label", lambda x: (x == "unclear").sum())
    )
    .reset_index()
)
item_summary["unclear_rate"] = item_summary["unclear_count"] / item_summary["total"]
item_summary = item_summary.sort_values(["unclear_rate", "unclear_count"], ascending=False)
item_summary.to_csv(RESULTS_DIR / "unclear_rate_by_item.csv", index=False)

# Unclear rate by clip type
type_summary = (
    df.groupby("clip_type")
    .agg(
        total=("label", "count"),
        unclear_count=("label", lambda x: (x == "unclear").sum())
    )
    .reset_index()
)
type_summary["unclear_rate"] = type_summary["unclear_count"] / type_summary["total"]
type_summary.to_csv(RESULTS_DIR / "unclear_rate_by_clip_type.csv", index=False)

print("\nOverall label counts:")
print(overall)

print("\nUnclear rate by sound:")
print(sound_summary)

print("\nUnclear rate by clip type:")
print(type_summary)

print("\nTop 10 hardest words/sentences:")
print(item_summary.head(10))

# Plot top 10 hardest items
top10 = item_summary.head(10).copy()

plt.figure(figsize=(10, 6))
plt.barh(top10["word"], top10["unclear_rate"])
plt.xlabel("Unclear Rate")
plt.ylabel("Word / Sentence")
plt.title("Top 10 Highest Unclear Rates")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "top10_unclear_items.png")
plt.close()

# Plot unclear rate by sound
plt.figure(figsize=(6, 4))
plt.bar(sound_summary["target_sound"], sound_summary["unclear_rate"])
plt.xlabel("Target Sound")
plt.ylabel("Unclear Rate")
plt.title("Unclear Rate by Target Sound")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "unclear_rate_by_sound.png")
plt.close()

# Plot unclear rate by clip type
plt.figure(figsize=(6, 4))
plt.bar(type_summary["clip_type"], type_summary["unclear_rate"])
plt.xlabel("Clip Type")
plt.ylabel("Unclear Rate")
plt.title("Unclear Rate: Words vs Sentences")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "unclear_rate_by_clip_type.png")
plt.close()

print("\nSaved analysis files in results/")
