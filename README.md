# Predicting Human Annotator Disagreement on CIFAR-10H

This project predicts full human annotator label distributions for CIFAR-10 images using CIFAR-10H.

## Best Model

CIFAR-adapted ResNet-18 pretrained on CIFAR-10 hard labels and fine-tuned on CIFAR-10H soft labels using KL divergence.

## Main Results

The best model achieved:

- Test KL: 0.2703
- Test JSD: 0.0556
- Cosine similarity: 0.9341
- Pearson entropy correlation: 0.4200
- Spearman entropy correlation: 0.4184
- Precision@100: 0.190
- Precision@200: 0.305
- Precision@500: 0.516

## How to Run

1. Run data sanity checks:
```bash
<<<<<<< HEAD
python src/01_data_sanity_checks.py
=======
python src/01_data_sanity_checks.py
>>>>>>> 516abfb297cf3bf266fb973e47cac5cd91b04ed7
