import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from dataset import get_dataloaders
from models import CIFARResNet18, count_parameters


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# -----------------------------
# Loss functions
# -----------------------------
def kl_loss_from_probs(p, q):
    eps = 1e-12
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)

    return torch.sum(p * torch.log(p / q), dim=1).mean()


def jsd_loss_from_logits(logits, target_probs):
    pred_probs = torch.softmax(logits, dim=1)

    eps = 1e-12
    target_probs = torch.clamp(target_probs, min=eps)
    pred_probs = torch.clamp(pred_probs, min=eps)

    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
    pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)

    m = 0.5 * (target_probs + pred_probs)

    jsd = 0.5 * kl_loss_from_probs(target_probs, m) + 0.5 * kl_loss_from_probs(pred_probs, m)

    return jsd


def entropy_from_probs(probs):
    eps = 1e-12
    return -torch.sum(probs * torch.log2(probs + eps), dim=1)


# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, soft_labels, hard_labels, entropies, indices in loader:
        images = images.to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = jsd_loss_from_logits(logits, soft_labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
    model.eval()

    total_jsd = 0.0
    total_entropy_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, soft_labels, hard_labels, entropies, indices in loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            entropies = entropies.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            jsd = jsd_loss_from_logits(logits, soft_labels)

            pred_entropy = entropy_from_probs(probs)
            entropy_mae = torch.mean(torch.abs(pred_entropy - entropies))

            batch_size = images.size(0)

            total_jsd += jsd.item() * batch_size
            total_entropy_mae += entropy_mae.item() * batch_size
            total_samples += batch_size

    avg_jsd = total_jsd / total_samples
    avg_entropy_mae = total_entropy_mae / total_samples

    return avg_jsd, avg_entropy_mae


# -----------------------------
# Plot curves
# -----------------------------
def plot_training_curves(train_losses, val_losses, val_entropy_mae):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train JSD Loss")
    plt.plot(epochs, val_losses, label="Validation JSD Loss")
    plt.xlabel("Epoch")
    plt.ylabel("JSD Loss")
    plt.title("JSD Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "jsd_training_validation_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_entropy_mae, label="Validation Entropy MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy MAE")
    plt.title("Validation Entropy Error Across Epochs - JSD")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "jsd_validation_entropy_mae.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


# -----------------------------
# Main
# -----------------------------
def main():
    print("Project root:", PROJECT_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    batch_size = 64
    num_workers = 2
    max_epochs = 30
    patience = 6

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=batch_size,
        num_workers=num_workers
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

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    val_entropy_mae_curve = []

    best_model_path = os.path.join(MODELS_DIR, "resnet18_jsd_best.pth")

    print("\nStarting JSD training...")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_jsd, val_entropy_mae = evaluate(model, val_loader, device)

        scheduler.step(val_jsd)

        train_losses.append(train_loss)
        val_losses.append(val_jsd)
        val_entropy_mae_curve.append(val_entropy_mae)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{max_epochs}] "
            f"Train JSD: {train_loss:.4f} | "
            f"Val JSD: {val_jsd:.4f} | "
            f"Val Entropy MAE: {val_entropy_mae:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_jsd < best_val_loss:
            best_val_loss = val_jsd
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_jsd": best_val_loss,
                    "epoch": epoch,
                    "head_type": "linear",
                    "loss": "JSD",
                    "seed": SEED
                },
                best_model_path
            )

            print("Saved best model:", best_model_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    plot_training_curves(train_losses, val_losses, val_entropy_mae_curve)

    print("\nTraining completed.")
    print("Best validation JSD:", best_val_loss)
    print("Best model saved at:", best_model_path)


if __name__ == "__main__":
    main()