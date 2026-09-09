# SpeakClear V2 Experiment Summary

## Main Finding

The best unclear-class F1 on the held-out SpeakClear test set was:

- Experiment: SpeakClear Only
- Model: SVM
- Accuracy: 0.738
- Balanced accuracy: 0.755
- Macro F1: 0.694
- Unclear precision: 0.458
- Unclear recall: 0.786
- Unclear F1: 0.579

## Interpretation

SpeechOcean public pronunciation data improved some models, especially Random Forest.
However, the best SpeakClear-only SVM remained slightly stronger by unclear-class F1.
This suggests that external pronunciation data is useful, but direct transfer is limited by dataset/domain mismatch.

## Recommended Paper Claim

Public expert-scored pronunciation data improved the experimental scope and model comparison, but did not completely replace the value of the personalized SpeakClear dataset.
The strongest result came from comparing personalized training, public-only transfer, and combined public-plus-personalized training on the same held-out SpeakClear test set.
