import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

from dataset import get_dataloaders
from models import CIFARResNet18


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Metric functions
# -----------------------------
def entropy_np(probs):
    eps = 1e-12
    return -np.sum(probs * np.log2(probs + eps), axis=1)


def kl_divergence_np(p, q):
    eps = 1e-12
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=1)


def js_divergence_np(p, q):
    eps = 1e-12

    p = p + eps
    q = q + eps

    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)

    return 0.5 * (kl_pm + kl_qm)


def cosine_similarity_per_sample(p, q):
    values = []

    for i in range(len(p)):
        sim = cosine_similarity(
            p[i].reshape(1, -1),
            q[i].reshape(1, -1)
        )[0][0]
        values.append(sim)

    return np.array(values)


def precision_at_k(true_entropy, pred_entropy, k):
    true_top_k = set(np.argsort(true_entropy)[-k:])
    pred_top_k = set(np.argsort(pred_entropy)[-k:])

    common = true_top_k.intersection(pred_top_k)

    return len(common) / k


# -----------------------------
# Collect predictions
# -----------------------------
def collect_predictions(model, loader, device):
    model.eval()

    all_true_probs = []
    all_pred_probs = []
    all_hard_labels = []
    all_indices = []

    with torch.no_grad():
        for images, soft_labels, hard_labels, entropies, indices in loader:
            images = images.to(device)

            logits = model(images)
            pred_probs = torch.softmax(logits, dim=1)

            all_true_probs.append(soft_labels.cpu().numpy())
            all_pred_probs.append(pred_probs.cpu().numpy())
            all_hard_labels.append(hard_labels.cpu().numpy())
            all_indices.append(indices.cpu().numpy())

    true_probs = np.concatenate(all_true_probs, axis=0)
    pred_probs = np.concatenate(all_pred_probs, axis=0)
    hard_labels = np.concatenate(all_hard_labels, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    return true_probs, pred_probs, hard_labels, indices


# -----------------------------
# Plot entropy scatter
# -----------------------------
def plot_entropy_scatter(true_entropy, pred_entropy):
    plt.figure(figsize=(7, 6))
    plt.scatter(true_entropy, pred_entropy, alpha=0.5)
    plt.xlabel("True Human Entropy")
    plt.ylabel("Predicted Entropy")
    plt.title("Predicted vs True Entropy - JSD Model")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "jsd_predicted_vs_true_entropy.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    _, _, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=64,
        num_workers=2
    )

    model = CIFARResNet18(
        num_classes=10,
        head_type="linear",
        pretrained=False
    ).to(device)

    model_path = os.path.join(MODELS_DIR, "resnet18_jsd_best.pth")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded model:", model_path)
    print("Best validation JSD:", checkpoint["best_val_jsd"])
    print("Best epoch:", checkpoint["epoch"])

    true_probs, pred_probs, hard_labels, indices = collect_predictions(
        model,
        test_loader,
        device
    )

    true_entropy = entropy_np(true_probs)
    pred_entropy = entropy_np(pred_probs)

    kl_values = kl_divergence_np(true_probs, pred_probs)
    jsd_values = js_divergence_np(true_probs, pred_probs)
    cosine_values = cosine_similarity_per_sample(true_probs, pred_probs)

    pearson_corr, pearson_p = pearsonr(true_entropy, pred_entropy)
    spearman_corr, spearman_p = spearmanr(true_entropy, pred_entropy)

    p_at_100 = precision_at_k(true_entropy, pred_entropy, 100)
    p_at_200 = precision_at_k(true_entropy, pred_entropy, 200)
    p_at_500 = precision_at_k(true_entropy, pred_entropy, 500)

    results = {
        "model": "ResNet18_JSD",
        "test_KL_mean": kl_values.mean(),
        "test_KL_std": kl_values.std(),
        "test_JSD_mean": jsd_values.mean(),
        "test_JSD_std": jsd_values.std(),
        "test_cosine_mean": cosine_values.mean(),
        "test_cosine_std": cosine_values.std(),
        "pearson_entropy_corr": pearson_corr,
        "spearman_entropy_corr": spearman_corr,
        "precision_at_100": p_at_100,
        "precision_at_200": p_at_200,
        "precision_at_500": p_at_500
    }

    print("\nTest Results")
    for key, value in results.items():
        print(f"{key}: {value}")

    results_df = pd.DataFrame([results])
    save_csv = os.path.join(RESULTS_DIR, "jsd_test_results.csv")
    results_df.to_csv(save_csv, index=False)

    print("\nSaved results:", save_csv)

    plot_entropy_scatter(true_entropy, pred_entropy)

    pred_save_path = os.path.join(RESULTS_DIR, "jsd_test_predictions.npz")
    np.savez(
        pred_save_path,
        indices=indices,
        hard_labels=hard_labels,
        true_probs=true_probs,
        pred_probs=pred_probs,
        true_entropy=true_entropy,
        pred_entropy=pred_entropy,
        kl_values=kl_values,
        jsd_values=jsd_values,
        cosine_values=cosine_values
    )

    print("Saved prediction details:", pred_save_path)


if __name__ == "__main__":
    main()