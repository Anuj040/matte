"""module for training refinement model"""
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

from model import MattingRefine
from utils.generator import DataGenerator

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pylint: disable = too-many-arguments, no-member, too-many-locals
class FineMatte:
    """class for building, training refined matte generator model"""

    def __init__(self) -> None:
        self.model = MattingRefine("resnet50").to(DEVICE)

    def generators(self) -> DataGenerator:
        """method to prepare and return generator objects in one place"""
        model_type = "refine"
        train_set = DataGenerator(model_type=model_type)
        train_generator = train_set(
            shuffle=True, batch_size=2, num_workers=8, pin_memory=True
        )
        valid_set = DataGenerator(
            dataset="alphamatting", mode="valid", model_type=model_type
        )
        valid_generator = valid_set()
        return train_generator, valid_generator

    def train(self):
        """model train method"""
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
        if not os.path.exists("checkpoint/matting_refine"):
            os.makedirs("checkpoint/matting_refine")
        writer = SummaryWriter("log/matting_refine")

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


if __name__ == "__main__":
    matte = FineMatte()
    matte.train()
