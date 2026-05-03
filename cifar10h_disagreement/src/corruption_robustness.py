import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from models import CIFARResNet18


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def normalize_image(x):
    return (x - CIFAR_MEAN) / CIFAR_STD


def entropy_from_probs(probs):
    eps = 1e-12
    return -torch.sum(probs * torch.log2(probs + eps), dim=1)


def add_gaussian_noise(images, severity):
    # severity controls standard deviation of noise
    noise_std = severity * 0.05
    noisy = images + torch.randn_like(images) * noise_std
    return torch.clamp(noisy, 0.0, 1.0)


def reduce_contrast(images, severity):
    # severity 0 = unchanged, severity 5 = strong contrast reduction
    factor = 1.0 - severity * 0.15
    factor = max(factor, 0.1)

    mean = images.mean(dim=(2, 3), keepdim=True)
    adjusted = mean + factor * (images - mean)

    return torch.clamp(adjusted, 0.0, 1.0)


def apply_gaussian_blur(images, severity):
    # Simple average blur approximation using avg_pool2d.
    # Kernel grows with severity.
    if severity == 0:
        return images

    kernel_size = 2 * severity + 1
    padding = severity

    blurred = torch.nn.functional.avg_pool2d(
        images,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    return torch.clamp(blurred, 0.0, 1.0)


def corrupt_images(images, corruption_type, severity):
    if severity == 0:
        return images

    if corruption_type == "gaussian_noise":
        return add_gaussian_noise(images, severity)

    if corruption_type == "gaussian_blur":
        return apply_gaussian_blur(images, severity)

    if corruption_type == "contrast_reduction":
        return reduce_contrast(images, severity)

    raise ValueError("Unknown corruption type.")


class CIFAR10HRawImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.cifar10 = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=False,
            transform=transforms.ToTensor()
        )

        probs_path = os.path.join(data_dir, "cifar10h-probs.npy")
        entropy_path = os.path.join(data_dir, "cifar10h_entropy.npy")

        self.soft_labels = np.load(probs_path).astype(np.float32)
        self.true_entropy = np.load(entropy_path).astype(np.float32)
        self.hard_labels = self.cifar10.targets

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, hard_label = self.cifar10[idx]

        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        true_entropy = torch.tensor(self.true_entropy[idx], dtype=torch.float32)
        hard_label = torch.tensor(hard_label, dtype=torch.long)

        return image, soft_label, hard_label, true_entropy, idx


def load_test_indices():
    test_path = os.path.join(DATA_DIR, "splits", "test_indices.npy")

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            "test_indices.npy not found. Run 02_dataloader_test.py first."
        )

    return np.load(test_path)


def evaluate_corruption(model, loader, device, corruption_type, severity):
    model.eval()

    all_pred_entropy = []
    all_true_entropy = []

    with torch.no_grad():
        for images, soft_labels, hard_labels, true_entropy, indices in loader:
            images = images.to(device)

            corrupted = corrupt_images(images, corruption_type, severity)

            # Model expects normalized images.
            corrupted = corrupted.cpu()
            corrupted = normalize_image(corrupted)
            corrupted = corrupted.to(device)

            logits = model(corrupted)
            probs = torch.softmax(logits, dim=1)

            pred_entropy = entropy_from_probs(probs)

            all_pred_entropy.append(pred_entropy.cpu().numpy())
            all_true_entropy.append(true_entropy.numpy())

    all_pred_entropy = np.concatenate(all_pred_entropy)
    all_true_entropy = np.concatenate(all_true_entropy)

    return {
        "corruption_type": corruption_type,
        "severity": severity,
        "mean_pred_entropy": float(all_pred_entropy.mean()),
        "std_pred_entropy": float(all_pred_entropy.std()),
        "mean_true_entropy": float(all_true_entropy.mean()),
        "entropy_gap": float(all_pred_entropy.mean() - all_true_entropy.mean())
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = CIFAR10HRawImageDataset(DATA_DIR)
    test_indices = load_test_indices()
    test_set = Subset(dataset, test_indices)

    test_loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    model = CIFARResNet18(
        num_classes=10,
        head_type="linear",
        pretrained=False
    ).to(device)

    model_path = os.path.join(
        MODELS_DIR,
        "resnet18_pretrained_finetuned_kl_best.pth"
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded best model:", model_path)

    corruption_types = [
        "gaussian_noise",
        "gaussian_blur",
        "contrast_reduction"
    ]

    severities = [0, 1, 2, 3, 4, 5]

    rows = []

    for corruption_type in corruption_types:
        print("\nEvaluating:", corruption_type)

        for severity in severities:
            result = evaluate_corruption(
                model=model,
                loader=test_loader,
                device=device,
                corruption_type=corruption_type,
                severity=severity
            )

            rows.append(result)

            print(
                f"Severity {severity} | "
                f"Mean Pred Entropy: {result['mean_pred_entropy']:.4f} | "
                f"Entropy Gap: {result['entropy_gap']:.4f}"
            )

    df = pd.DataFrame(rows)

    save_csv = os.path.join(RESULTS_DIR, "corruption_robustness_best_model.csv")
    df.to_csv(save_csv, index=False)

    print("\nSaved:", save_csv)

    # Plot corruption response
    plt.figure(figsize=(9, 6))

    for corruption_type in corruption_types:
        sub = df[df["corruption_type"] == corruption_type]
        plt.plot(
            sub["severity"],
            sub["mean_pred_entropy"],
            marker="o",
            label=corruption_type
        )

    clean_true_entropy = df[df["severity"] == 0]["mean_true_entropy"].iloc[0]
    plt.axhline(
        y=clean_true_entropy,
        linestyle="--",
        label="Mean true entropy"
    )

    plt.xlabel("Corruption Severity")
    plt.ylabel("Mean Predicted Entropy")
    plt.title("Predicted Entropy under Image Corruptions")
    plt.legend()
    plt.tight_layout()

    save_plot = os.path.join(PLOTS_DIR, "corruption_response_entropy_best_model.png")
    plt.savefig(save_plot, dpi=300)
    plt.close()

    print("Saved:", save_plot)

    # Separate entropy gap plot
    plt.figure(figsize=(9, 6))

    for corruption_type in corruption_types:
        sub = df[df["corruption_type"] == corruption_type]
        plt.plot(
            sub["severity"],
            sub["entropy_gap"],
            marker="o",
            label=corruption_type
        )

    plt.axhline(y=0.0, linestyle="--")
    plt.xlabel("Corruption Severity")
    plt.ylabel("Predicted Entropy - True Entropy")
    plt.title("Entropy Gap under Corruptions")
    plt.legend()
    plt.tight_layout()

    save_plot = os.path.join(PLOTS_DIR, "corruption_entropy_gap_best_model.png")
    plt.savefig(save_plot, dpi=300)
    plt.close()

    print("Saved:", save_plot)


if __name__ == "__main__":
    main()