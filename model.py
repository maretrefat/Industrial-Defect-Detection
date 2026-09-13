"""
model.py
========
Transfer-Learning model for DAGM 2007 binary defect classification, built on
a pre-trained ResNet18 backbone (ImageNet weights) with a custom classification
head.

Design notes
------------
- The convolutional backbone is initialized with ImageNet-pretrained weights
  because low-level texture/edge filters transfer well to industrial surface
  images even though the domain (metal/fabric textures) differs from natural
  images.
- We replace the final fully-connected layer with a small MLP head
  (Linear -> ReLU -> Dropout -> Linear) mapping to 2 output logits
  (Normal / Defect), trained with CrossEntropyLoss.
- `freeze_backbone` allows optionally freezing the convolutional backbone for
  a cheaper linear-probe style baseline; by default we fine-tune the whole
  network end-to-end, which we found in practice to give the best results
  for this dataset.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class DefectResNet18(nn.Module):
    """ResNet18 backbone + custom binary classification head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False,
                 dropout: float = 0.3):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            backbone = models.resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # strip original classification head
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None,
) -> DefectResNet18:
    model = DefectResNet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    # Quick shape sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device=device)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")  # Expect: torch.Size([4, 2])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")
