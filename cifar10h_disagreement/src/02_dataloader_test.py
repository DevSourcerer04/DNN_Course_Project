import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------
# Basic paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


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
# Custom CIFAR-10H Dataset
# -----------------------------
class CIFAR10HDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=False,
            download=False,
            transform=transform
        )

        probs_path = os.path.join(root, "cifar10h-probs.npy")
        entropy_path = os.path.join(root, "cifar10h_entropy.npy")

        if not os.path.exists(probs_path):
            raise FileNotFoundError(
                "cifar10h-probs.npy not found. Run 01_data_sanity_checks.py first."
            )

        if not os.path.exists(entropy_path):
            raise FileNotFoundError(
                "cifar10h_entropy.npy not found. Run 01_data_sanity_checks.py first."
            )

        self.soft_labels = np.load(probs_path).astype(np.float32)
        self.entropies = np.load(entropy_path).astype(np.float32)
        self.hard_labels = self.cifar10.targets

        assert len(self.cifar10) == 10000
        assert self.soft_labels.shape == (10000, 10)
        assert self.entropies.shape == (10000,)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, hard_label = self.cifar10[idx]

        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        entropy = torch.tensor(self.entropies[idx], dtype=torch.float32)
        hard_label = torch.tensor(hard_label, dtype=torch.long)

        return image, soft_label, hard_label, entropy, idx


# -----------------------------
# Create fixed train/val/test split
# -----------------------------
def create_splits(dataset_size=10000):
    indices = np.arange(dataset_size)

    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    train_indices = indices[:6000]
    val_indices = indices[6000:8000]
    test_indices = indices[8000:10000]

    split_dir = os.path.join(DATA_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)

    np.save(os.path.join(split_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(split_dir, "val_indices.npy"), val_indices)
    np.save(os.path.join(split_dir, "test_indices.npy"), test_indices)

    return train_indices, val_indices, test_indices


# -----------------------------
# Main
# -----------------------------
def main():
    print("Project root:", PROJECT_ROOT)
    print("Data dir:", DATA_DIR)

    # Basic transform for now.
    # We normalize using CIFAR-10 mean/std because ResNet training becomes more stable.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = CIFAR10HDataset(
        root=DATA_DIR,
        transform=transform
    )

    train_indices, val_indices, test_indices = create_splits(len(dataset))

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("\nSplit sizes")
    print("Train:", len(train_set))
    print("Validation:", len(val_set))
    print("Test:", len(test_set))

    images, soft_labels, hard_labels, entropies, indices = next(iter(train_loader))

    print("\nOne batch check")
    print("Images shape:", images.shape)
    print("Soft labels shape:", soft_labels.shape)
    print("Hard labels shape:", hard_labels.shape)
    print("Entropies shape:", entropies.shape)
    print("Indices shape:", indices.shape)

    print("\nSoft label example:")
    print(soft_labels[0])
    print("Soft label sum:", soft_labels[0].sum().item())
    print("Hard label:", hard_labels[0].item())
    print("Entropy:", entropies[0].item())

    print("\nDataLoader test completed successfully.")


if __name__ == "__main__":
    main()