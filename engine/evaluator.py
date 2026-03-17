from typing import Dict

import torch
from tqdm import tqdm

from utils.mask_utils import dice_score


@torch.no_grad()
def evaluate_generator(generator, dataloader, device) -> Dict[str, float]:
    generator.eval()
    dice_values = []
    mse_values = []
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = generator(
            batch["prior"], batch["current"], batch["breast_mask"], batch["view_id"], batch["side_id"]
        )
        mse = torch.mean((outputs["synthetic"] - batch["current"]) ** 2).item()
        dice = dice_score(outputs["tumor_map"], batch["tumor_mask"]).item()
        mse_values.append(mse)
        dice_values.append(dice)
    return {
        "mse": float(sum(mse_values) / max(len(mse_values), 1)),
        "dice": float(sum(dice_values) / max(len(dice_values), 1)),
    }
