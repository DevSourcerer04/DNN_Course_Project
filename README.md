# Predicting Human Annotator Disagreement on CIFAR-10H

## Project Overview

This Deep Neural Networks course project predicts the full human annotator label distribution for CIFAR-10 images using CIFAR-10H. Unlike a standard hard-label classifier, the model is designed to predict how humans are likely to disagree across the 10 CIFAR-10 classes.

Given a 32×32 CIFAR-10 image, the model outputs a 10-dimensional probability distribution representing the expected distribution of human annotator responses.

## Objective

The objective is to model human uncertainty and disagreement directly, rather than predicting only one majority-vote class. This allows the model to capture ambiguity in images where human annotators may reasonably assign different labels.

## Datasets

| Dataset | Description |
|---|---|
| CIFAR-10 | 60,000 images across 10 classes, with 50,000 training images and 10,000 test images. |
| CIFAR-10H | Human annotator label distributions for the 10,000 CIFAR-10 test images. |

The CIFAR-10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

For CIFAR-10H experiments, the 10,000 examples were split using fixed seed 42:

| Split | Size |
|---|---:|
| Train | 6,000 |
| Validation | 2,000 |
| Test | 2,000 |

## Best Model

The best-performing model was a CIFAR-adapted ResNet-18 with:

- 3x3 first convolution with stride 1
- Removed maxpool layer
- Linear prediction head
- CIFAR-10 hard-label pretraining
- CIFAR-10H soft-label fine-tuning
- KL Divergence loss

Standard ResNet-18 is designed for 224×224 ImageNet images, but CIFAR-10 images are only 32×32. Therefore, the original 7×7 stride-2 first convolution was replaced by a 3×3 stride-1 convolution, and the maxpool layer was removed. This preserves spatial detail that would otherwise be lost too early for small CIFAR-10 images.

## Main Results

| Metric | Value |
|---|---:|
| Test KL Divergence | 0.2703 |
| Test Jensen-Shannon Divergence | 0.0556 |
| Test Cosine Similarity | 0.9341 |
| Pearson Entropy Correlation | 0.4200 |
| Spearman Entropy Correlation | 0.4184 |
| Precision@100 | 0.190 |
| Precision@200 | 0.305 |
| Precision@500 | 0.516 |

## Loss Functions Compared

The following loss functions were compared:

- KL Divergence
- Jensen-Shannon Divergence
- Custom KL + Entropy Loss

The custom loss was defined as:

```text
Custom Loss = KL(true distribution || predicted distribution) + λ × |True Entropy - Predicted Entropy|
```

where λ = 0.5.

Overall, KL Divergence performed best across the main evaluation metrics.

## Ablation Studies

| Ablation | Comparison | Finding |
|---|---|---|
| Loss function comparison | KL, JSD, custom KL + entropy | KL performed best overall. |
| Prediction head comparison | Linear head vs MLP head | The linear prediction head performed better. |
| Training strategy comparison | Soft-label only vs CIFAR-10 hard-label pretraining followed by CIFAR-10H fine-tuning | Pretraining followed by fine-tuning performed much better. |

## Robustness Checks

Class-wise performance showed that the model worked better on visually distinct classes such as automobile, truck, ship, and frog. It struggled more on ambiguous animal classes such as bird, cat, dog, and deer.

For corruption robustness, Gaussian noise, Gaussian blur, and contrast reduction were applied. Predicted entropy increased as corruption severity increased, especially for Gaussian blur. This suggests that the model's uncertainty response is meaningfully related to image degradation.

## Explainability and Analysis

Failure case analysis was conducted using:

- Highest KL divergence cases
- Highest entropy error cases
- Over-uncertain cases
- Over-confident cases

Manual disagreement source analysis used the following categories:

- ambiguous_object_identity
- poor_image_quality
- multi_object_or_multilabel
- boundary_between_classes
- similar_looking_classes
- background_or_context_confusion
- other

Grad-CAM analysis showed that low-disagreement images usually focus on the main object. High-disagreement images often show diffuse focus or attention on ambiguous regions and contextual background areas.

## Project Structure

```text
cifar10h_disagreement/
├── data/
├── outputs/
│   ├── models/
│   ├── plots/
│   └── results/
├── src/
│   ├── 01_data_sanity_checks.py
│   ├── 02_dataloader_test.py
│   ├── dataset.py
│   ├── models.py
│   ├── train_kl.py
│   ├── evaluate_kl.py
│   ├── train_jsd.py
│   ├── evaluate_jsd.py
│   ├── train_custom.py
│   ├── evaluate_custom.py
│   ├── compare_results.py
│   ├── train_kl_mlp.py
│   ├── evaluate_kl_mlp.py
│   ├── compare_head_ablation.py
│   ├── pretrain_cifar10.py
│   ├── finetune_pretrained_kl.py
│   ├── evaluate_pretrained_finetuned_kl.py
│   ├── classwise_analysis.py
│   ├── corruption_robustness.py
│   ├── failure_case_analysis.py
│   ├── manual_disagreement_analysis.py
│   ├── fill_manual_disagreement_csv.py
│   └── gradcam_analysis.py
├── report.tex
├── report.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Create and activate the Conda environment:

```bash
conda create -n dnn_cifar10h python=3.10 -y
conda activate dnn_cifar10h
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn scipy tqdm pillow opencv-python grad-cam
```

Alternatively, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run

Run the scripts from inside the `cifar10h_disagreement/` directory:

```bash
python src/01_data_sanity_checks.py
python src/02_dataloader_test.py
python src/train_kl.py
python src/evaluate_kl.py
python src/train_jsd.py
python src/evaluate_jsd.py
python src/train_custom.py
python src/evaluate_custom.py
python src/compare_results.py
python src/train_kl_mlp.py
python src/evaluate_kl_mlp.py
python src/compare_head_ablation.py
python src/pretrain_cifar10.py
python src/finetune_pretrained_kl.py
python src/evaluate_pretrained_finetuned_kl.py
python src/classwise_analysis.py
python src/corruption_robustness.py
python src/failure_case_analysis.py
python src/manual_disagreement_analysis.py
python src/fill_manual_disagreement_csv.py
python src/gradcam_analysis.py
```

## Important Outputs

Best model:

```text
outputs/models/resnet18_pretrained_finetuned_kl_best.pth
```

Important result files:

```text
outputs/results/all_loss_comparison.csv
outputs/results/head_ablation_comparison.csv
outputs/results/pretrained_finetuned_kl_test_results.csv
outputs/results/classwise_performance_best_model.csv
outputs/results/corruption_robustness_best_model.csv
outputs/results/failure_cases_best_model.csv
outputs/results/manual_disagreement_source_filled.csv
```

Important plots:

```text
outputs/plots/entropy_histogram.png
outputs/plots/human_confusion_matrix.png
outputs/plots/loss_function_grouped_comparison.png
outputs/plots/classwise_kl_best_model.png
outputs/plots/corruption_response_entropy_best_model.png
outputs/plots/failure_cases_highest_kl.png
outputs/plots/manual_high_entropy_image_grid.png
outputs/plots/gradcam_low_disagreement_examples.png
outputs/plots/gradcam_high_disagreement_examples.png
```

## Final Conclusion

This project shows that human annotator disagreement is meaningful and can be modeled directly. The best approach was CIFAR-10 hard-label pretraining followed by CIFAR-10H soft-label fine-tuning using KL Divergence.

## References

- Peterson et al., *Human uncertainty makes classification more robust*, ICCV 2019.
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Selvaraju et al., *Grad-CAM*, ICCV 2017.
