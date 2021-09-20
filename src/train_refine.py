"""module for training refinement model"""
import datetime
import os

import kornia
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from src.model import MattingRefine
from src.utils.augmentation import random_crop, train_step_augmenter
from src.utils.generator import define_generators

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pylint: disable = too-many-locals, too-many-statements
class FineMatte:
    """class for building, training refined matte generator model"""

    def __init__(self) -> None:
        self.model = MattingRefine("resnet50").to(DEVICE)

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

        train_loader, valid_loader = define_generators(
            "refine", batch_size, num_workers=num_workers
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
                true_pha, true_fgr, true_bgr = random_crop(
                    1024, true_pha, true_fgr, true_bgr
                )

                true_src = true_bgr.clone()

                # TODO: # Augment with shadow

                # Composite foreground onto source
                true_src = true_fgr * true_pha + true_src * (1 - true_pha)

                # Implement train step augmentations
                true_src, true_bgr = train_step_augmenter(true_src, true_bgr)

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

                if (i + 1) % int(10 * 2 / batch_size) == 0:
                    writer.add_scalar("loss", loss, step)

                if (i + 1) % int(90 * 2 / batch_size) == 0:
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
