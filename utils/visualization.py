from pathlib import Path

import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def save_triplet(prior, current, synthetic, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 4))
    for idx, image in enumerate([prior, current, synthetic], start=1):
        ax = fig.add_subplot(1, 3, idx)
        ax.imshow(image.squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
