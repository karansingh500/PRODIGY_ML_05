import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(num_classes: int = 101, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    for param in model.features.parameters():
        param.requires_grad = not freeze


def save_checkpoint(path, model, class_names, metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, checkpoint.get("metrics", {})
