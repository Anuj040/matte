"""module for training model base"""
import os
import random

import kornia
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm

from model import MattingBase
from utils.generator import DataGenerator

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pylint: disable = no-member, too-many-arguments, too-many-locals


class CoarseMatte:
    """class for building, training coarse matte generator model"""

    def __init__(self) -> None:
        self.model = MattingBase("resnet50").to(DEVICE)

    def generators(self) -> DataGenerator:
        """method to prepare and return generator objects in one place"""
        train_set = DataGenerator()
        train_generator = train_set(
            shuffle=True, batch_size=2, num_workers=8, pin_memory=True
        )
        valid_set = DataGenerator(dataset="alphamatting", mode="valid")
        valid_generator = valid_set()
        return train_generator, valid_generator

    def train(self):
        """model train method"""
        optimizer = Adam(
            [
                {"params": self.model.backbone.parameters(), "lr": 1e-4},
                {"params": self.model.aspp.parameters(), "lr": 5e-4},
                {"params": self.model.decoder.parameters(), "lr": 5e-4},
            ]
        )
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Logging and checkpoints
        if not os.path.exists("checkpoint/matting_base"):
            os.makedirs("checkpoint/matting_base")
        writer = SummaryWriter("log/matting_base")

        train_loader, valid_loader = self.generators()

        # Run loop
        for epoch in range(0, 10):
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
                    pred_pha, pred_fgr, pred_err = self.model(true_src, true_bgr)[:3]
                    loss = compute_loss(
                        pred_pha, pred_fgr, pred_err, true_pha, true_fgr
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
                        "train_pred_err", make_grid(pred_err, nrow=5), step
                    )
                    writer.add_image(
                        "train_true_src", make_grid(true_src, nrow=5), step
                    )
                    writer.add_image(
                        "train_true_bgr", make_grid(true_bgr, nrow=5), step
                    )

                del true_pha, true_fgr, true_bgr
                del pred_pha, pred_fgr, pred_err

        #     if (i + 1) % args.log_valid_interval == 0:
        #         valid(model, dataloader_valid, writer, step)

        #     if step % args.checkpoint_interval == 0:
        #         torch.save(
        #             model.state_dict(),
        #             f"checkpoint/{args.model_name}/epoch-{epoch}-iter-{step}.pth",
        #         )

        # torch.save(
        #     model.state_dict(), f"checkpoint/{args.model_name}/epoch-{epoch}.pth"
        # )


# --------------- Utils ---------------


def random_crop(*imgs):
    """method to take matching random crop out of the image set"""
    width = random.choice(range(256, 512))
    height = random.choice(range(256, 512))
    results = []
    for img in imgs:
        img = kornia.resize(img, (max(height, width), max(height, width)))
        img = kornia.center_crop(img, (height, width))
        results.append(img)
    return results


def compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr):
    """loss calculating function"""
    true_err = torch.abs(pred_pha.detach() - true_pha)
    true_msk = true_pha != 0
    return (
        F.l1_loss(pred_pha, true_pha)
        + F.l1_loss(kornia.sobel(pred_pha), kornia.sobel(true_pha))
        + F.l1_loss(pred_fgr * true_msk, true_fgr * true_msk)
        + F.mse_loss(pred_err, true_err)
    )


if __name__ == "__main__":
    matte = CoarseMatte()
    matte.train()
