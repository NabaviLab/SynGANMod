import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from configs.default_config import ExperimentConfig
from data.dataset import LongitudinalMammogramDataset
from engine.trainer import Trainer
from losses.adversarial_loss import AdversarialLoss
from losses.total_loss import CompositeGeneratorLoss, LossWeights
from models.discriminator import SwinDiscriminator
from models.generator import ProjectionAwareTumorGenerator
from utils.logger import build_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/default")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    logger = build_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LongitudinalMammogramDataset(args.data_root, args.train_csv, image_size=cfg.data.image_size)
    val_ds = LongitudinalMammogramDataset(args.data_root, args.val_csv, image_size=cfg.data.image_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)

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
    discriminator = SwinDiscriminator(cfg.model.swin_name, cfg.model.swin_pretrained).to(device)

    g_optimizer = Adam(generator.parameters(), lr=cfg.train.learning_rate_g, betas=(cfg.train.beta1, cfg.train.beta2), weight_decay=cfg.train.weight_decay)
    d_optimizer = Adam(discriminator.parameters(), lr=cfg.train.learning_rate_d, betas=(cfg.train.beta1, cfg.train.beta2), weight_decay=cfg.train.weight_decay)

    g_criterion = CompositeGeneratorLoss(
        max_area_fraction=cfg.model.max_area_fraction,
        weights=LossWeights(
            lambda_kl=cfg.train.lambda_kl,
            lambda_adv=cfg.train.lambda_adv,
            lambda_tumor=cfg.train.lambda_tumor,
            lambda_intensity=cfg.train.lambda_intensity,
            lambda_area=cfg.train.lambda_area,
        ),
    )
    d_criterion = AdversarialLoss()

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_criterion=g_criterion,
        d_criterion=d_criterion,
        device=device,
        output_dir=args.output_dir,
        logger=logger,
        amp=cfg.train.amp,
        grad_clip=cfg.train.grad_clip,
        save_every=cfg.train.save_every,
    )
    trainer.train(train_loader, val_loader, cfg.train.epochs)


if __name__ == "__main__":
    main()
