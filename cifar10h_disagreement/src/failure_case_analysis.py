import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def kl_divergence_np(p, q):
    eps = 1e-12
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=1)


def entropy_np(probs):
    eps = 1e-12
    return -np.sum(probs * np.log2(probs + eps), axis=1)


def short_distribution_text(probs, top_k=3):
    top_indices = np.argsort(probs)[::-1][:top_k]

    parts = []
    for idx in top_indices:
        parts.append(f"{CLASS_NAMES[idx]}:{probs[idx]:.2f}")

    return "\n".join(parts)


def plot_failure_grid(
    cifar_dataset,
    selected_rows,
    title,
    save_name
):
    num_examples = len(selected_rows)

    fig, axes = plt.subplots(
        num_examples,
        3,
        figsize=(12, 3.2 * num_examples)
    )

    if num_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(selected_rows):
        img_idx = int(row["image_index"])

        image, hard_label = cifar_dataset[img_idx]

        true_probs = row["true_probs"]
        pred_probs = row["pred_probs"]

        # Column 1: image
        ax = axes[row_idx, 0]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"Image idx: {img_idx}\n"
            f"CIFAR label: {CLASS_NAMES[hard_label]}\n"
            f"KL: {row['kl']:.3f}"
        )

        # Column 2: true human distribution
        ax = axes[row_idx, 1]
        ax.bar(CLASS_NAMES, true_probs)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(
            f"True human distribution\n"
            f"True H: {row['true_entropy']:.3f}\n"
            f"{short_distribution_text(true_probs)}"
        )

        # Column 3: predicted distribution
        ax = axes[row_idx, 2]
        ax.bar(CLASS_NAMES, pred_probs)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(
            f"Predicted distribution\n"
            f"Pred H: {row['pred_entropy']:.3f}\n"
            f"{short_distribution_text(pred_probs)}"
        )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def main():
    pred_path = os.path.join(
        RESULTS_DIR,
        "pretrained_finetuned_kl_test_predictions.npz"
    )

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            "Prediction file not found. Run evaluate_pretrained_finetuned_kl.py first."
        )

    data = np.load(pred_path)

    indices = data["indices"]
    hard_labels = data["hard_labels"]
    true_probs = data["true_probs"]
    pred_probs = data["pred_probs"]

    true_entropy = entropy_np(true_probs)
    pred_entropy = entropy_np(pred_probs)

    kl_values = kl_divergence_np(true_probs, pred_probs)
    entropy_abs_error = np.abs(true_entropy - pred_entropy)

    rows = []

    for i in range(len(indices)):
        rows.append({
            "rank_position": i,
            "image_index": int(indices[i]),
            "hard_label": int(hard_labels[i]),
            "hard_label_name": CLASS_NAMES[int(hard_labels[i])],
            "true_entropy": float(true_entropy[i]),
            "pred_entropy": float(pred_entropy[i]),
            "entropy_abs_error": float(entropy_abs_error[i]),
            "kl": float(kl_values[i]),
            "true_top_class": int(np.argmax(true_probs[i])),
            "true_top_class_name": CLASS_NAMES[int(np.argmax(true_probs[i]))],
            "pred_top_class": int(np.argmax(pred_probs[i])),
            "pred_top_class_name": CLASS_NAMES[int(np.argmax(pred_probs[i]))],
            "true_probs": true_probs[i],
            "pred_probs": pred_probs[i]
        })

    df = pd.DataFrame(rows)

    # Save CSV without array columns for readability
    csv_df = df.drop(columns=["true_probs", "pred_probs"])
    save_csv = os.path.join(RESULTS_DIR, "failure_cases_best_model.csv")
    csv_df.to_csv(save_csv, index=False)

    print("\nSaved failure case table:", save_csv)

    # Load original PIL images
    cifar_test = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=None
    )

    # Top failures by KL divergence
    top_kl = df.sort_values("kl", ascending=False).head(6)
    top_kl_rows = top_kl.to_dict("records")

    print("\nTop KL failure cases:")
    print(top_kl.drop(columns=["true_probs", "pred_probs"]).to_string(index=False))

    plot_failure_grid(
        cifar_dataset=cifar_test,
        selected_rows=top_kl_rows,
        title="Failure Cases: Highest KL Divergence",
        save_name="failure_cases_highest_kl.png"
    )

    # Top failures by entropy error
    top_entropy_error = df.sort_values("entropy_abs_error", ascending=False).head(6)
    top_entropy_rows = top_entropy_error.to_dict("records")

    print("\nTop entropy-error failure cases:")
    print(top_entropy_error.drop(columns=["true_probs", "pred_probs"]).to_string(index=False))

    plot_failure_grid(
        cifar_dataset=cifar_test,
        selected_rows=top_entropy_rows,
        title="Failure Cases: Highest Entropy Error",
        save_name="failure_cases_highest_entropy_error.png"
    )

    # Under-confident cases: predicted entropy much higher than true entropy
    df["entropy_signed_error"] = df["pred_entropy"] - df["true_entropy"]
    over_uncertain = df.sort_values("entropy_signed_error", ascending=False).head(6)
    over_uncertain_rows = over_uncertain.to_dict("records")

    print("\nMost over-uncertain cases:")
    print(over_uncertain.drop(columns=["true_probs", "pred_probs"]).to_string(index=False))

    plot_failure_grid(
        cifar_dataset=cifar_test,
        selected_rows=over_uncertain_rows,
        title="Failure Cases: Model Overestimates Disagreement",
        save_name="failure_cases_over_uncertain.png"
    )

    # Over-confident cases: predicted entropy much lower than true entropy
    over_confident = df.sort_values("entropy_signed_error", ascending=True).head(6)
    over_confident_rows = over_confident.to_dict("records")

    print("\nMost over-confident cases:")
    print(over_confident.drop(columns=["true_probs", "pred_probs"]).to_string(index=False))

    plot_failure_grid(
        cifar_dataset=cifar_test,
        selected_rows=over_confident_rows,
        title="Failure Cases: Model Underestimates Disagreement",
        save_name="failure_cases_over_confident.png"
    )


if __name__ == "__main__":
    main()