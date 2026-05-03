import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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


def create_or_load_splits(data_dir, dataset_size=10000):
    split_dir = os.path.join(data_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, "train_indices.npy")
    val_path = os.path.join(split_dir, "val_indices.npy")
    test_path = os.path.join(split_dir, "test_indices.npy")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_indices = np.load(train_path)
        val_indices = np.load(val_path)
        test_indices = np.load(test_path)
        return train_indices, val_indices, test_indices

    random.seed(SEED)
    np.random.seed(SEED)

    indices = np.arange(dataset_size)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    train_indices = indices[:6000]
    val_indices = indices[6000:8000]
    test_indices = indices[8000:10000]

    np.save(train_path, train_indices)
    np.save(val_path, val_indices)
    np.save(test_path, test_indices)

    return train_indices, val_indices, test_indices


def get_dataloaders(data_dir, batch_size=64, num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_full = CIFAR10HDataset(root=data_dir, transform=train_transform)
    eval_full = CIFAR10HDataset(root=data_dir, transform=eval_transform)

    train_indices, val_indices, test_indices = create_or_load_splits(data_dir)

    train_set = Subset(train_full, train_indices)
    val_set = Subset(eval_full, val_indices)
    test_set = Subset(eval_full, test_indices)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader