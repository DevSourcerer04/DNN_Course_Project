import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms


# -----------------------------
# Basic paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# CIFAR-10 class names
# -----------------------------
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# -----------------------------
# Load / create CIFAR-10H probability vectors
# -----------------------------
def load_or_create_cifar10h_probs():
    csv_path = os.path.join(DATA_DIR, "cifar10h-raw.csv")
    npy_path = os.path.join(DATA_DIR, "cifar10h-probs.npy")

    if os.path.exists(npy_path):
        print("Loading existing CIFAR-10H probabilities:", npy_path)
        return np.load(npy_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Could not find cifar10h-raw.csv inside the data folder."
        )

    print("Loading CIFAR-10H raw CSV:", csv_path)
    df = pd.read_csv(csv_path)

    print("CSV shape before filtering:", df.shape)
    print("CSV columns:", df.columns.tolist())

    # Remove attention-check rows.
    # These are not real CIFAR-10 images and may have index -99999.
    if "is_attn_check" in df.columns:
        df = df[df["is_attn_check"] == 0].copy()

    # Keep only valid CIFAR-10 test image rows.
    df = df[df["cifar10_test_test_idx"] >= 0].copy()
    df = df[df["cifar10_test_test_idx"] < 10000].copy()

    print("CSV shape after filtering:", df.shape)

    probs = np.zeros((10000, 10), dtype=np.float32)

    for _, row in df.iterrows():
        img_idx = int(row["cifar10_test_test_idx"])
        label = int(row["chosen_label"])

        if 0 <= label <= 9:
            probs[img_idx, label] += 1.0

    row_sums = probs.sum(axis=1, keepdims=True)

    zero_vote_images = np.where(row_sums.squeeze() == 0)[0]
    print("Images with zero votes:", len(zero_vote_images))

    if len(zero_vote_images) > 0:
        raise ValueError(
            "Some CIFAR-10 images have zero votes after filtering. "
            "This should not happen for the full CIFAR-10H raw file."
        )

    probs = probs / row_sums

    np.save(npy_path, probs)
    print("Saved converted probabilities:", npy_path)

    return probs


# -----------------------------
# Entropy function
# -----------------------------
def compute_entropy(probs):
    eps = 1e-12
    return -np.sum(probs * np.log2(probs + eps), axis=1)


# -----------------------------
# Plot entropy histogram
# -----------------------------
def plot_entropy_histogram(entropies):
    plt.figure(figsize=(8, 5))
    plt.hist(entropies, bins=40, edgecolor="black")
    plt.xlabel("Human Label Entropy")
    plt.ylabel("Number of Images")
    plt.title("CIFAR-10H Entropy Distribution")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "entropy_histogram.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


# -----------------------------
# Plot average entropy per class
# -----------------------------
def plot_per_class_entropy(entropies, hard_labels):
    hard_labels = np.array(hard_labels)
    class_avg_entropy = []

    for class_id in range(10):
        idx = np.where(hard_labels == class_id)[0]
        class_avg_entropy.append(entropies[idx].mean())

    plt.figure(figsize=(10, 5))
    plt.bar(CLASS_NAMES, class_avg_entropy)
    plt.xticks(rotation=45)
    plt.ylabel("Average Entropy")
    plt.title("Average Human Disagreement per CIFAR-10 Class")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "per_class_entropy.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


# -----------------------------
# Human confusion-style matrix
# -----------------------------
def plot_human_confusion_matrix(probs, hard_labels):
    hard_labels = np.array(hard_labels)
    matrix = np.zeros((10, 10))

    for true_class in range(10):
        idx = np.where(hard_labels == true_class)[0]
        matrix[true_class] = probs[idx].mean(axis=0)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix)
    plt.colorbar(label="Average human probability")
    plt.xticks(range(10), CLASS_NAMES, rotation=45)
    plt.yticks(range(10), CLASS_NAMES)
    plt.xlabel("Human chosen label")
    plt.ylabel("Original CIFAR-10 label")
    plt.title("Human Annotator Distribution Matrix")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "human_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


# -----------------------------
# Low/high entropy examples
# -----------------------------
def plot_low_high_entropy_examples(dataset, probs, entropies):
    sorted_indices = np.argsort(entropies)

    low_indices = sorted_indices[:8]
    high_indices = sorted_indices[-8:]
    selected_indices = list(low_indices) + list(high_indices)

    plt.figure(figsize=(16, 6))

    for plot_idx, img_idx in enumerate(selected_indices):
        image, hard_label = dataset[img_idx]

        plt.subplot(2, 8, plot_idx + 1)
        plt.imshow(image)
        plt.axis("off")

        human_top_class = int(np.argmax(probs[img_idx]))

        title = (
            f"CIFAR:{CLASS_NAMES[hard_label]}\n"
            f"Human:{CLASS_NAMES[human_top_class]}\n"
            f"H:{entropies[img_idx]:.2f}"
        )

        plt.title(title, fontsize=8)

    plt.suptitle("Top row: low disagreement | Bottom row: high disagreement")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "low_high_entropy_examples.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


# -----------------------------
# Main
# -----------------------------
def main():
    print("Project root:", PROJECT_ROOT)
    print("Data dir:", DATA_DIR)

    transform = transforms.ToTensor()

    # download=False because CIFAR-10 is already manually present in:
    # data/cifar-10-batches-py/
    cifar10_test_tensor = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform
    )

    cifar10_test_pil = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=None
    )

    hard_labels = cifar10_test_tensor.targets

    probs = load_or_create_cifar10h_probs()

    print("\nDataset checks")
    print("CIFAR-10 test images:", len(cifar10_test_tensor))
    print("CIFAR-10H probs shape:", probs.shape)

    assert probs.shape == (10000, 10), (
        "CIFAR-10H probabilities should be shape (10000, 10)."
    )

    row_sums = probs.sum(axis=1)
    print("Minimum probability sum:", row_sums.min())
    print("Maximum probability sum:", row_sums.max())
    print("Mean probability sum:", row_sums.mean())

    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        "Some probability rows do not sum to 1."
    )

    entropies = compute_entropy(probs)

    print("\nEntropy statistics")
    print("Minimum entropy:", entropies.min())
    print("Maximum entropy:", entropies.max())
    print("Mean entropy:", entropies.mean())
    print("Std entropy:", entropies.std())

    np.save(os.path.join(DATA_DIR, "cifar10h_entropy.npy"), entropies)

    plot_entropy_histogram(entropies)
    plot_per_class_entropy(entropies, hard_labels)
    plot_human_confusion_matrix(probs, hard_labels)
    plot_low_high_entropy_examples(cifar10_test_pil, probs, entropies)

    print("\nData sanity checks completed successfully.")


if __name__ == "__main__":
    main()