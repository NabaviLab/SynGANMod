import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinDiscriminator(nn.Module):
    def __init__(self, model_name: str = "swin_tiny_patch4_window7_224", pretrained: bool = True):
        super().__init__()
        self.input_adapter = nn.Conv2d(3, 3, kernel_size=1)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.classifier = nn.Linear(self.backbone.num_features, 1)

    def forward(self, prior: torch.Tensor, current: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prior, current, candidate], dim=1)
        x = self.input_adapter(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.backbone(x)
        return torch.sigmoid(self.classifier(feat))
