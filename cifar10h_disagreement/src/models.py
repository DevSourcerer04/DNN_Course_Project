import torch
import torch.nn as nn
import torchvision.models as models


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10, head_type="linear", pretrained=False):
        super(CIFARResNet18, self).__init__()

        # We start from torchvision ResNet-18 structure.
        # pretrained=False by default because ImageNet weights are designed for large images.
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # CIFAR-10 adaptation:
        # Original ResNet uses 7x7 conv stride 2, which is too aggressive for 32x32 images.
        # We replace it with 3x3 conv stride 1 to preserve spatial information.
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove maxpool because CIFAR-10 images are already tiny.
        self.backbone.maxpool = nn.Identity()

        feature_dim = self.backbone.fc.in_features

        # Remove original classification layer.
        self.backbone.fc = nn.Identity()

        # Prediction head.
        # It outputs logits. Softmax will be applied during loss/evaluation.
        if head_type == "linear":
            self.head = nn.Linear(feature_dim, num_classes)

        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )

        else:
            raise ValueError("head_type must be either 'linear' or 'mlp'.")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CIFARResNet18(
        num_classes=10,
        head_type="linear",
        pretrained=False
    ).to(device)

    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    output = model(dummy_input)

    total_params, trainable_params = count_parameters(model)

    print("Device:", device)
    print("Output shape:", output.shape)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    probs = torch.softmax(output, dim=1)
    print("Probability shape:", probs.shape)
    print("Probability sum for first sample:", probs[0].sum().item())