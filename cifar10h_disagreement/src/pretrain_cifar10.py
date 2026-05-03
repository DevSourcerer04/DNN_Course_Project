import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import CIFARResNet18, count_parameters


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            preds = torch.argmax(logits, dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train CE Loss")
    plt.plot(epochs, val_losses, label="Validation CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("CIFAR-10 Hard-Label Pretraining Loss")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "cifar10_pretraining_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Hard-Label Pretraining Accuracy")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "cifar10_pretraining_accuracy.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


def main():
    print("Project root:", PROJECT_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    batch_size = 128
    num_workers = 2
    max_epochs = 20
    patience = 5

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

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    model = CIFARResNet18(
        num_classes=10,
        head_type="linear",
        pretrained=False
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_model_path = os.path.join(MODELS_DIR, "resnet18_cifar10_pretrained.pth")

    print("\nStarting CIFAR-10 hard-label pretraining...")

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{max_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "head_type": "linear",
                    "pretraining": "CIFAR10_HARD_LABELS",
                    "seed": SEED
                },
                best_model_path
            )

            print("Saved best pretrained model:", best_model_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    plot_curves(train_losses, val_losses, train_accs, val_accs)

    print("\nPretraining completed.")
    print("Best validation accuracy:", best_val_acc)
    print("Best pretrained model saved at:", best_model_path)


if __name__ == "__main__":
    main()