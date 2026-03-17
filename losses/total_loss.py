from dataclasses import dataclass

import torch
import torch.nn as nn

from .adversarial_loss import AdversarialLoss
from .area_loss import AreaRegularizationLoss
from .intensity_loss import IntensityConsistencyLoss
from .kl_loss import KLDivergenceLoss
from .reconstruction_loss import ReconstructionLoss
from .tumor_loss import TumorLocalizationLoss


@dataclass
class LossWeights:
    lambda_kl: float = 0.1
    lambda_adv: float = 1.0
    lambda_tumor: float = 5.0
    lambda_intensity: float = 10.0
    lambda_area: float = 2.0


class CompositeGeneratorLoss(nn.Module):
    def __init__(self, max_area_fraction: float, weights: LossWeights):
        super().__init__()
        self.recon = ReconstructionLoss()
        self.kl = KLDivergenceLoss()
        self.adv = AdversarialLoss()
        self.tumor = TumorLocalizationLoss()
        self.intensity = IntensityConsistencyLoss()
        self.area = AreaRegularizationLoss(max_area_fraction)
        self.weights = weights

    def forward(self, outputs, batch, fake_pred):
        recon = self.recon(outputs["synthetic"], batch["current"])
        adv = self.adv.g_loss(fake_pred)
        intensity = self.intensity(outputs["synthetic"], batch["current"], outputs["blend_mask"])
        has_tumor = batch["has_tumor_mask"].float().view(-1, 1, 1, 1)

        tumor = self.tumor(outputs["tumor_map"], batch["tumor_mask"])
        tumor = (tumor * has_tumor.mean())
        kl = self.kl(outputs["mu"], outputs["logvar"])
        area = self.area(outputs["blend_mask"])

        normal_mask = 1.0 - has_tumor.mean()
        total = recon + self.weights.lambda_adv * adv + self.weights.lambda_intensity * intensity
        total = total + has_tumor.mean() * (
            self.weights.lambda_kl * kl
            + self.weights.lambda_tumor * tumor
            + self.weights.lambda_area * area
        )
        return total, {
            "recon": recon.detach(),
            "adv": adv.detach(),
            "intensity": intensity.detach(),
            "tumor": tumor.detach(),
            "kl": kl.detach(),
            "area": area.detach(),
            "normal_factor": normal_mask.detach(),
        }
