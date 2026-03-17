from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from engine.evaluator import evaluate_generator
from utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, g_criterion, d_criterion, device, output_dir, logger, amp=True, grad_clip=1.0, save_every=5):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_criterion = g_criterion
        self.d_criterion = d_criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.amp = amp
        self.grad_clip = grad_clip
        self.save_every = save_every
        self.scaler = GradScaler(enabled=amp)
        self.best_mse = float("inf")

    def _move(self, batch):
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}

    def train(self, train_loader, val_loader, epochs: int):
        for epoch in range(1, epochs + 1):
            self.generator.train()
            self.discriminator.train()
            epoch_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in progress:
                batch = self._move(batch)
                with autocast(enabled=self.amp):
                    outputs = self.generator(batch["prior"], batch["current"], batch["breast_mask"], batch["view_id"], batch["side_id"])
                    real_pred = self.discriminator(batch["prior"], batch["current"], batch["current"])
                    fake_pred_detached = self.discriminator(batch["prior"], batch["current"], outputs["synthetic"].detach())
                    d_loss = self.d_criterion.d_loss(real_pred, fake_pred_detached)

                self.d_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(d_loss).backward()
                self.scaler.unscale_(self.d_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                self.scaler.step(self.d_optimizer)

                with autocast(enabled=self.amp):
                    fake_pred = self.discriminator(batch["prior"], batch["current"], outputs["synthetic"])
                    g_loss, loss_dict = self.g_criterion(outputs, batch, fake_pred)

                self.g_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(g_loss).backward()
                self.scaler.unscale_(self.g_optimizer)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
                self.scaler.step(self.g_optimizer)
                self.scaler.update()

                epoch_loss += g_loss.item()
                progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

            metrics = evaluate_generator(self.generator, val_loader, self.device)
            avg_loss = epoch_loss / max(len(train_loader), 1)
            self.logger.info(
                "Epoch %d | train_g_loss=%.4f | val_mse=%.4f | val_dice=%.4f",
                epoch,
                avg_loss,
                metrics["mse"],
                metrics["dice"],
            )
            if epoch % self.save_every == 0:
                self._save(epoch, metrics, is_best=False)
            if metrics["mse"] < self.best_mse:
                self.best_mse = metrics["mse"]
                self._save(epoch, metrics, is_best=True)

    def _save(self, epoch, metrics, is_best=False):
        suffix = "best" if is_best else f"epoch_{epoch}"
        save_checkpoint({"model": self.generator.state_dict(), "epoch": epoch, "metrics": metrics}, str(self.output_dir / f"generator_{suffix}.pt"))
        save_checkpoint({"model": self.discriminator.state_dict(), "epoch": epoch, "metrics": metrics}, str(self.output_dir / f"discriminator_{suffix}.pt"))
