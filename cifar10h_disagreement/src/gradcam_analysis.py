import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import CIFARResNet18


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_normalized_tensor(pil_image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])

    return transform(pil_image).unsqueeze(0)


def pil_to_float_image(pil_image):
    image = np.array(pil_image).astype(np.float32) / 255.0
    return image


def entropy_np(probs):
    eps = 1e-12
    return -np.sum(probs * np.log2(probs + eps), axis=1)


def top_distribution_text(probs, top_k=3):
    top_indices = np.argsort(probs)[::-1][:top_k]

    parts = []
    for idx in top_indices:
        parts.append(f"{CLASS_NAMES[idx]}:{probs[idx]:.2f}")

    return "\n".join(parts)


def create_gradcam_grid(
    model,
    cam,
    cifar_test,
    selected_indices,
    true_probs_all,
    pred_probs_all,
    true_entropy_all,
    pred_entropy_all,
    title,
    save_name,
    device
):
    num_examples = len(selected_indices)

    fig, axes = plt.subplots(
        num_examples,
        3,
        figsize=(12, 3.2 * num_examples)
    )

    if num_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, img_idx in enumerate(selected_indices):
        pil_image, hard_label = cifar_test[int(img_idx)]

        raw_float_image = pil_to_float_image(pil_image)
        input_tensor = get_normalized_tensor(pil_image).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = int(torch.argmax(logits, dim=1).item())

        targets = [ClassifierOutputTarget(pred_class)]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        cam_overlay = show_cam_on_image(
            raw_float_image,
            grayscale_cam,
            use_rgb=True
        )

        true_probs = true_probs_all[row_idx]
        pred_probs = pred_probs_all[row_idx]

        # Column 1: original image
        axes[row_idx, 0].imshow(pil_image)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(
            f"Original\n"
            f"Idx: {img_idx}\n"
            f"CIFAR: {CLASS_NAMES[hard_label]}"
        )

        # Column 2: Grad-CAM overlay
        axes[row_idx, 1].imshow(cam_overlay)
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title(
            f"Grad-CAM\n"
            f"Target: {CLASS_NAMES[pred_class]}"
        )

        # Column 3: distribution comparison as text
        axes[row_idx, 2].axis("off")
        text = (
            f"True H: {true_entropy_all[row_idx]:.3f}\n"
            f"Pred H: {pred_entropy_all[row_idx]:.3f}\n\n"
            f"True top:\n{top_distribution_text(true_probs)}\n\n"
            f"Pred top:\n{top_distribution_text(pred_probs)}"
        )
        axes[row_idx, 2].text(
            0.02,
            0.5,
            text,
            fontsize=10,
            va="center"
        )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    pred_path = os.path.join(
        RESULTS_DIR,
        "pretrained_finetuned_kl_test_predictions.npz"
    )

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            "Prediction file not found. Run evaluate_pretrained_finetuned_kl.py first."
        )

    data = np.load(pred_path)

    test_indices = data["indices"]
    true_probs = data["true_probs"]
    pred_probs = data["pred_probs"]
    true_entropy = data["true_entropy"]
    pred_entropy = data["pred_entropy"]

    cifar_test = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=None
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
    model.eval()

    print("Loaded best model:", model_path)

    # We use the final convolutional block of ResNet-18.
    # Reason: Grad-CAM works best on the last convolutional layer because it has
    # high-level semantic features while still preserving spatial information.
    target_layers = [model.backbone.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # Select 5 lowest and 5 highest true-entropy test examples.
    low_positions = np.argsort(true_entropy)[:5]
    high_positions = np.argsort(true_entropy)[-5:][::-1]

    low_image_indices = test_indices[low_positions]
    high_image_indices = test_indices[high_positions]

    print("\nLow-disagreement selected image indices:")
    print(low_image_indices)

    print("\nHigh-disagreement selected image indices:")
    print(high_image_indices)

    create_gradcam_grid(
        model=model,
        cam=cam,
        cifar_test=cifar_test,
        selected_indices=low_image_indices,
        true_probs_all=true_probs[low_positions],
        pred_probs_all=pred_probs[low_positions],
        true_entropy_all=true_entropy[low_positions],
        pred_entropy_all=pred_entropy[low_positions],
        title="Grad-CAM Analysis: Low-Disagreement Images",
        save_name="gradcam_low_disagreement_examples.png",
        device=device
    )

    create_gradcam_grid(
        model=model,
        cam=cam,
        cifar_test=cifar_test,
        selected_indices=high_image_indices,
        true_probs_all=true_probs[high_positions],
        pred_probs_all=pred_probs[high_positions],
        true_entropy_all=true_entropy[high_positions],
        pred_entropy_all=pred_entropy[high_positions],
        title="Grad-CAM Analysis: High-Disagreement Images",
        save_name="gradcam_high_disagreement_examples.png",
        device=device
    )

    print("\nGrad-CAM analysis completed.")


if __name__ == "__main__":
    main()