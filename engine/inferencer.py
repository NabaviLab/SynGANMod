from pathlib import Path

import torch
from tqdm import tqdm

from utils.visualization import save_triplet


@torch.no_grad()
def run_inference(generator, dataloader, device, save_dir: str):
    generator.eval()
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(dataloader, desc="Inference"):
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = generator(
            batch["prior"], batch["current"], batch["breast_mask"], batch["view_id"], batch["side_id"]
        )
        for i, case_id in enumerate(batch["case_id"]):
            save_triplet(
                batch["prior"][i],
                batch["current"][i],
                outputs["synthetic"][i],
                str(save_root / f"{case_id}.png"),
            )
