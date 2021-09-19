"""module for training refinement model"""
import datetime
import os
import random
from typing import Tuple

import kornia
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm

from src.model import MattingRefine
from src.utils.generator import DataGenerator

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pylint: disable = too-many-arguments, no-member, too-many-locals, too-many-statements
class FineMatte:
    """class for building, training refined matte generator model"""

    def __init__(self) -> None:
        self.model = MattingRefine("resnet50").to(DEVICE)

    def generators(
        self, batch_size: int = 2, num_workers: int = 8
    ) -> Tuple[DataLoader]:
        """method to prepare and return generator objects in one place

        Args:
            batch_size (int, optional): Defaults to 2.
            num_workers (int, optional): Number of cpu workers for generators.
        """

        model_type = "refine"
        train_set = DataGenerator(model_type=model_type)
        train_generator = train_set(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        valid_set = DataGenerator(
            dataset="alphamatting", mode="valid", model_type=model_type
        )
        valid_generator = valid_set(batch_size=batch_size, num_workers=num_workers)
        return train_generator, valid_generator

    def train(
        self, epochs: int = 10, batch_size: int = 2, num_workers: int = 8
    ) -> None:
        """model train method

        Args:
            epochs (int, optional): Training epochs. Defaults to 10.
            batch_size (int, optional): Defaults to 2.
            num_workers (int, optional): Number of cpu workers for generators.
        """

        optimizer = Adam(
            [
                {"params": self.model.backbone.parameters(), "lr": 5e-5},
                {"params": self.model.aspp.parameters(), "lr": 5e-5},
                {"params": self.model.decoder.parameters(), "lr": 1e-4},
                {"params": self.model.refiner.parameters(), "lr": 3e-4},
            ]
        )
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Logging and checkpoints
        now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        if not os.path.exists(f"checkpoint/matting_refine/{now}"):
            os.makedirs(f"checkpoint/matting_refine/{now}")
        writer = SummaryWriter(f"log/matting_refine/{now}")

        train_loader, valid_loader = self.generators(
            batch_size, num_workers=num_workers
        )
        # Initialize validation loss
        valid_loss = 1e9
        # Run loop
        for epoch in range(0, epochs):
            for i, ((true_pha, true_fgr), true_bgr) in enumerate(tqdm(train_loader)):
                step = epoch * len(train_loader) + i + 1

                true_pha = true_pha.to(DEVICE)
                true_fgr = true_fgr.to(DEVICE)
                true_bgr = true_bgr.to(DEVICE)
                true_pha, true_fgr, true_bgr = random_crop(true_pha, true_fgr, true_bgr)

                true_src = true_bgr.clone()

                # TODO: # Augment with shadow

                # Composite foreground onto source
                true_src = true_fgr * true_pha + true_src * (1 - true_pha)

                # Augment with noise
                aug_noise_idx = torch.rand(len(true_src)) < 0.4
                if aug_noise_idx.any():
                    true_src[aug_noise_idx] = (
                        true_src[aug_noise_idx]
                        .add_(
                            torch.randn_like(true_src[aug_noise_idx]).mul_(
                                0.03 * random.random()
                            )
                        )
                        .clamp_(0, 1)
                    )
                    true_bgr[aug_noise_idx] = (
                        true_bgr[aug_noise_idx]
                        .add_(
                            torch.randn_like(true_bgr[aug_noise_idx]).mul_(
                                0.03 * random.random()
                            )
                        )
                        .clamp_(0, 1)
                    )
                del aug_noise_idx

                # Augment background with jitter
                aug_jitter_idx = torch.rand(len(true_src)) < 0.8
                if aug_jitter_idx.any():
                    true_bgr[aug_jitter_idx] = kornia.augmentation.ColorJitter(
                        0.18, 0.18, 0.18, 0.1
                    )(true_bgr[aug_jitter_idx])
                del aug_jitter_idx

                # Augment background with affine
                aug_affine_idx = torch.rand(len(true_bgr)) < 0.3
                if aug_affine_idx.any():
                    true_bgr[aug_affine_idx] = T.RandomAffine(
                        degrees=(-1, 1), translate=(0.01, 0.01)
                    )(true_bgr[aug_affine_idx])
                del aug_affine_idx

                with autocast(enabled=torch.cuda.is_available()):
                    (
                        pred_pha,
                        pred_fgr,
                        pred_pha_sm,
                        pred_fgr_sm,
                        pred_err_sm,
                        _,
                    ) = self.model(true_src, true_bgr)
                    loss = compute_loss(
                        pred_pha,
                        pred_fgr,
                        pred_pha_sm,
                        pred_fgr_sm,
                        pred_err_sm,
                        true_pha,
                        true_fgr,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

                if (i + 1) % 10 == 0:
                    writer.add_scalar("loss", loss, step)

                if (i + 1) % 90 == 0:
                    writer.add_image(
                        "train_pred_pha", make_grid(pred_pha, nrow=5), step
                    )
                    writer.add_image(
                        "train_pred_fgr", make_grid(pred_fgr, nrow=5), step
                    )
                    writer.add_image(
                        "train_pred_com", make_grid(pred_fgr * pred_pha, nrow=5), step
                    )
                    writer.add_image(
                        "train_pred_err", make_grid(pred_err_sm, nrow=5), step
                    )
                    writer.add_image(
                        "train_true_src", make_grid(true_src, nrow=5), step
                    )

                del true_pha, true_fgr, true_src, true_bgr
                del pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm

            current_val_loss = valid(self.model, valid_loader, writer, step)

            if current_val_loss < valid_loss:
                valid_loss = current_val_loss
                torch.save(
                    self.model.state_dict(),
                    f"checkpoint/matting_refine/{now}/epoch-{epoch}-loss-{valid_loss:.4f}.pth",
                )


# --------------- Utils ---------------


def compute_loss(
    pred_pha_lg,
    pred_fgr_lg,
    pred_pha_sm,
    pred_fgr_sm,
    pred_err_sm,
    true_pha_lg,
    true_fgr_lg,
):
    """loss calculating function"""
    true_pha_sm = kornia.resize(true_pha_lg, pred_pha_sm.shape[2:])
    true_fgr_sm = kornia.resize(true_fgr_lg, pred_fgr_sm.shape[2:])
    true_msk_lg = true_pha_lg != 0
    true_msk_sm = true_pha_sm != 0
    return (
        F.l1_loss(pred_pha_lg, true_pha_lg)
        + F.l1_loss(pred_pha_sm, true_pha_sm)
        + F.l1_loss(kornia.sobel(pred_pha_lg), kornia.sobel(true_pha_lg))
        + F.l1_loss(kornia.sobel(pred_pha_sm), kornia.sobel(true_pha_sm))
        + F.l1_loss(pred_fgr_lg * true_msk_lg, true_fgr_lg * true_msk_lg)
        + F.l1_loss(pred_fgr_sm * true_msk_sm, true_fgr_sm * true_msk_sm)
        + F.mse_loss(
            kornia.resize(pred_err_sm, true_pha_lg.shape[2:]),
            kornia.resize(pred_pha_sm, true_pha_lg.shape[2:]).sub(true_pha_lg).abs(),
        )
    )


def random_crop(*imgs):
    """method to take matching random crop out of the image set"""
    h_src, w_src = imgs[0].shape[2:]
    w_tgt = random.choice(range(1024, 2048)) // 4 * 4
    h_tgt = random.choice(range(1024, 2048)) // 4 * 4
    scale = max(w_tgt / w_src, h_tgt / h_src)
    results = []
    for img in imgs:
        img = kornia.resize(img, (int(h_src * scale), int(w_src * scale)))
        img = kornia.center_crop(img, (h_tgt, w_tgt))
        results.append(img)
    return results


def valid(
    model: nn.Module, dataloader: DataLoader, writer: SummaryWriter, step: int
) -> float:
    """model evaluation step executor

    Args:
        model (nn.Module): [description]
        dataloader (DataLoader): [description]
        writer (SummaryWriter): [description]
        step (int): [description]

    Returns:
        float: validation loss
    """

    model.eval()
    loss_total = 0
    loss_count = 0
    with torch.no_grad():
        for (true_pha, true_fgr), true_bgr in dataloader:
            batch_size = true_pha.size(0)

            true_pha = true_pha.to(DEVICE)
            true_fgr = true_fgr.to(DEVICE)
            true_bgr = true_bgr.to(DEVICE)
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, _ = model(
                true_src, true_bgr
            )
            loss = compute_loss(
                pred_pha,
                pred_fgr,
                pred_pha_sm,
                pred_fgr_sm,
                pred_err_sm,
                true_pha,
                true_fgr,
            )
            loss_total += loss.cpu().item() * batch_size
            loss_count += batch_size

    writer.add_scalar("valid_loss", loss_total / loss_count, step)
    model.train()
    return loss_total / loss_count


if __name__ == "__main__":
    matte = FineMatte()
    matte.train()
