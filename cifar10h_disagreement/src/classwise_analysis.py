import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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
    numerator = np.sum(p * q, axis=1)
    denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1)

    eps = 1e-12
    return numerator / (denominator + eps)


def safe_corr(x, y, method="pearson"):
    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    if method == "pearson":
        return pearsonr(x, y)[0]

    if method == "spearman":
        return spearmanr(x, y)[0]

    return np.nan


def main():
    pred_path = os.path.join(
        RESULTS_DIR,
        "pretrained_finetuned_kl_test_predictions.npz"
    )

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            "Best model prediction file not found. "
            "Run evaluate_pretrained_finetuned_kl.py first."
        )

    data = np.load(pred_path)

    true_probs = data["true_probs"]
    pred_probs = data["pred_probs"]
    hard_labels = data["hard_labels"]

    true_entropy = data["true_entropy"]
    pred_entropy = data["pred_entropy"]

    kl_values = kl_divergence_np(true_probs, pred_probs)
    jsd_values = js_divergence_np(true_probs, pred_probs)
    cosine_values = cosine_similarity_per_sample(true_probs, pred_probs)
    entropy_abs_error = np.abs(true_entropy - pred_entropy)

    rows = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = hard_labels == class_id

        class_true_entropy = true_entropy[mask]
        class_pred_entropy = pred_entropy[mask]

        row = {
            "class_id": class_id,
            "class_name": class_name,
            "num_test_images": int(mask.sum()),
            "true_entropy_mean": class_true_entropy.mean(),
            "pred_entropy_mean": class_pred_entropy.mean(),
            "entropy_mae": entropy_abs_error[mask].mean(),
            "kl_mean": kl_values[mask].mean(),
            "jsd_mean": jsd_values[mask].mean(),
            "cosine_mean": cosine_values[mask].mean(),
            "pearson_entropy_corr": safe_corr(
                class_true_entropy,
                class_pred_entropy,
                method="pearson"
            ),
            "spearman_entropy_corr": safe_corr(
                class_true_entropy,
                class_pred_entropy,
                method="spearman"
            )
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    save_csv = os.path.join(RESULTS_DIR, "classwise_performance_best_model.csv")
    df.to_csv(save_csv, index=False)

    print("\nClass-wise Performance")
    print(df.to_string(index=False))
    print("\nSaved:", save_csv)

    # Plot KL per class
    plt.figure(figsize=(10, 5))
    plt.bar(df["class_name"], df["kl_mean"])
    plt.xticks(rotation=45)
    plt.ylabel("Mean KL Divergence")
    plt.title("Class-wise KL Divergence - Best Model")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "classwise_kl_best_model.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)

    # Plot entropy MAE per class
    plt.figure(figsize=(10, 5))
    plt.bar(df["class_name"], df["entropy_mae"])
    plt.xticks(rotation=45)
    plt.ylabel("Entropy MAE")
    plt.title("Class-wise Entropy Prediction Error - Best Model")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "classwise_entropy_mae_best_model.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)

    # Plot true vs predicted average entropy per class
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, df["true_entropy_mean"], width, label="True Entropy")
    plt.bar(x + width / 2, df["pred_entropy_mean"], width, label="Predicted Entropy")
    plt.xticks(x, CLASS_NAMES, rotation=45)
    plt.ylabel("Mean Entropy")
    plt.title("True vs Predicted Mean Entropy per Class")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "classwise_true_vs_pred_entropy_best_model.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)

    # Plot cosine similarity per class
    plt.figure(figsize=(10, 5))
    plt.bar(df["class_name"], df["cosine_mean"])
    plt.xticks(rotation=45)
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Class-wise Distribution Similarity - Best Model")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "classwise_cosine_best_model.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()