from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("results_final")
OUT_DIR.mkdir(exist_ok=True)

speakclear = pd.read_csv("results_v2/speakclear_model_results_v2.csv")
speechocean = pd.read_csv("results_public/speechocean_model_results.csv")
transfer = pd.read_csv("results_transfer/transfer_model_results.csv")

speakclear["evaluation_context"] = "SpeakClear internal evaluation"
speechocean["evaluation_context"] = "SpeechOcean internal evaluation"
transfer["evaluation_context"] = "Transfer to held-out SpeakClear test set"

speakclear["experiment"] = speakclear.get("experiment", "SpeakClear Only")
speechocean["experiment"] = "SpeechOcean Only"
transfer["experiment"] = transfer["experiment"]

common_cols = [
    "evaluation_context",
    "experiment",
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "unclear_precision",
    "unclear_recall",
    "unclear_f1",
    "test_clear_support",
    "test_unclear_support",
]

combined = pd.concat(
    [
        speakclear[common_cols],
        speechocean[common_cols],
        transfer[common_cols],
    ],
    ignore_index=True,
)

combined.to_csv(OUT_DIR / "all_v2_model_results.csv", index=False)

# Transfer-only sorted table
transfer_sorted = transfer[[
    "experiment",
    "model",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "unclear_precision",
    "unclear_recall",
    "unclear_f1",
    "test_clear_support",
    "test_unclear_support",
]].sort_values(["unclear_f1", "balanced_accuracy"], ascending=False)

transfer_sorted.to_csv(OUT_DIR / "transfer_results_sorted.csv", index=False)

# Best model per evaluation context
best_rows = []
for context, group in combined.groupby("evaluation_context"):
    best = group.sort_values(["unclear_f1", "balanced_accuracy"], ascending=False).iloc[0]
    best_rows.append(best)

best_df = pd.DataFrame(best_rows)
best_df.to_csv(OUT_DIR / "best_models_summary.csv", index=False)

# Chart 1: Transfer unclear F1
plot_df = transfer.copy()
plot_df["label"] = plot_df["experiment"] + "\n" + plot_df["model"]

plt.figure(figsize=(12, 6))
plt.bar(plot_df["label"], plot_df["unclear_f1"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("Unclear-class F1")
plt.title("Transfer Experiment: Unclear-Class F1 on Held-Out SpeakClear Test Set")
plt.tight_layout()
plt.savefig(OUT_DIR / "transfer_unclear_f1_comparison.png", dpi=250)
plt.close()

# Chart 2: Transfer balanced accuracy
plt.figure(figsize=(12, 6))
plt.bar(plot_df["label"], plot_df["balanced_accuracy"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("Balanced accuracy")
plt.title("Transfer Experiment: Balanced Accuracy on Held-Out SpeakClear Test Set")
plt.tight_layout()
plt.savefig(OUT_DIR / "transfer_balanced_accuracy_comparison.png", dpi=250)
plt.close()

# Markdown summary
best_transfer = transfer_sorted.iloc[0]

summary = f"""# SpeakClear V2 Experiment Summary

## Main Finding

The best unclear-class F1 on the held-out SpeakClear test set was:

- Experiment: {best_transfer["experiment"]}
- Model: {best_transfer["model"]}
- Accuracy: {best_transfer["accuracy"]:.3f}
- Balanced accuracy: {best_transfer["balanced_accuracy"]:.3f}
- Macro F1: {best_transfer["macro_f1"]:.3f}
- Unclear precision: {best_transfer["unclear_precision"]:.3f}
- Unclear recall: {best_transfer["unclear_recall"]:.3f}
- Unclear F1: {best_transfer["unclear_f1"]:.3f}

## Interpretation

SpeechOcean public pronunciation data improved some models, especially Random Forest.
However, the best SpeakClear-only SVM remained slightly stronger by unclear-class F1.
This suggests that external pronunciation data is useful, but direct transfer is limited by dataset/domain mismatch.

## Recommended Paper Claim

Public expert-scored pronunciation data improved the experimental scope and model comparison, but did not completely replace the value of the personalized SpeakClear dataset.
The strongest result came from comparing personalized training, public-only transfer, and combined public-plus-personalized training on the same held-out SpeakClear test set.
"""

with open(OUT_DIR / "v2_experiment_summary.md", "w") as f:
    f.write(summary)

print("Created results_final files:")
for p in sorted(OUT_DIR.iterdir()):
    print("-", p)
