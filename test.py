import argparse

import torch
from torch.utils.data import DataLoader

from configs.default_config import ExperimentConfig
from data.dataset import LongitudinalMammogramDataset
from engine.evaluator import evaluate_generator
from models.generator import ProjectionAwareTumorGenerator
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = LongitudinalMammogramDataset(args.data_root, args.test_csv, image_size=cfg.data.image_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    generator = ProjectionAwareTumorGenerator(
        image_size=cfg.data.image_size,
        patch_size=cfg.data.patch_size,
        embed_dim=cfg.model.embed_dim,
        latent_dim=cfg.model.latent_dim,
        encoder_depth=cfg.model.encoder_depth,
        decoder_depth=cfg.model.decoder_depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
    ).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["model"])

    metrics = evaluate_generator(generator, test_loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
