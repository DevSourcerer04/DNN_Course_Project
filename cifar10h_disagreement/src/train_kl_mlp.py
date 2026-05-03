import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from dataset import get_dataloaders
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


def kl_loss_from_logits(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


def entropy_from_probs(probs):
    eps = 1e-12
    return -torch.sum(probs * torch.log2(probs + eps), dim=1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, soft_labels, hard_labels, entropies, indices in loader:
        images = images.to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = kl_loss_from_logits(logits, soft_labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, loader, device):
    model.eval()

    total_kl = 0.0
    total_entropy_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, soft_labels, hard_labels, entropies, indices in loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            entropies = entropies.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            kl = kl_loss_from_logits(logits, soft_labels)

            pred_entropy = entropy_from_probs(probs)
            entropy_mae = torch.mean(torch.abs(pred_entropy - entropies))

            batch_size = images.size(0)

            total_kl += kl.item() * batch_size
            total_entropy_mae += entropy_mae.item() * batch_size
            total_samples += batch_size

    return total_kl / total_samples, total_entropy_mae / total_samples


def plot_training_curves(train_losses, val_losses, val_entropy_mae):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train KL Loss")
    plt.plot(epochs, val_losses, label="Validation KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("KL Training and Validation Loss - MLP Head")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "kl_mlp_training_validation_loss.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_entropy_mae, label="Validation Entropy MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy MAE")
    plt.title("Validation Entropy Error - MLP Head")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "kl_mlp_validation_entropy_mae.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Saved:", save_path)


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
        head_type="mlp",
        pretrained=False
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print("Head type: MLP")
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

    best_model_path = os.path.join(MODELS_DIR, "resnet18_kl_mlp_best.pth")

    print("\nStarting KL training with MLP head...")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_kl, val_entropy_mae = evaluate(model, val_loader, device)

        scheduler.step(val_kl)

        train_losses.append(train_loss)
        val_losses.append(val_kl)
        val_entropy_mae_curve.append(val_entropy_mae)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{max_epochs}] "
            f"Train KL: {train_loss:.4f} | "
            f"Val KL: {val_kl:.4f} | "
            f"Val Entropy MAE: {val_entropy_mae:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_kl < best_val_loss:
            best_val_loss = val_kl
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_kl": best_val_loss,
                    "epoch": epoch,
                    "head_type": "mlp",
                    "loss": "KL",
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
    print("Best validation KL:", best_val_loss)
    print("Best model saved at:", best_model_path)


if __name__ == "__main__":
    main()