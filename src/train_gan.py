"""module for training refine matte model with GAN"""

import datetime
import os

import kornia
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from src.model import MattingRefine
from src.model.networks import NLayerDiscriminator
from src.utils.augmentation import random_crop, train_step_augmenter
from src.utils.generator import define_generators
from src.utils.train_utils import (
    compute_refine_loss,
    gan_loss,
    set_requires_grad,
    valid,
)

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pylint: disable = too-many-locals, too-many-statements
class GANMatte:
    """class for building, training refined matte generator model"""

    def __init__(self, n_layers: int = 1, gan_weight: float = 0.002) -> None:
        self.model = MattingRefine("resnet50").to(DEVICE)
        self.discrimintaor = NLayerDiscriminator(
            input_nc=4, n_layers=n_layers, norm_layer=nn.BatchNorm2d
        ).to(DEVICE)
        self.gan_weight = gan_weight

    def train(
        self, epochs: int = 10, batch_size: int = 2, num_workers: int = 8
    ) -> None:
        """model train method

        Args:
            epochs (int, optional): Training epochs. Defaults to 10.
            batch_size (int, optional): Defaults to 2.
            num_workers (int, optional): Number of cpu workers for generators.
        """
        # TODO check the impace of weight decay paramerter
        # weight_decay=0.0005
        g_optimizer = Adam(
            [
                {"params": self.model.backbone.parameters(), "lr": 5e-5},
                {"params": self.model.aspp.parameters(), "lr": 5e-5},
                {"params": self.model.decoder.parameters(), "lr": 1e-4},
                {"params": self.model.refiner.parameters(), "lr": 3e-4},
            ]
        )
        d_optimizer = Adam(self.discrimintaor.parameters(), lr=1e-4)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # Logging and checkpoints
        now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        if not os.path.exists(f"checkpoint/matting_gan/{now}"):
            os.makedirs(f"checkpoint/matting_gan/{now}")
        writer = SummaryWriter(f"log/matting_gan/{now}")

        train_loader, valid_loader = define_generators(
            "refine", batch_size, num_workers=num_workers
        )
        # Initialize validation loss
        valid_loss = 1e9
        # Run loop
        for epoch in range(0, epochs):

            self.model.train()
            for i, ((true_pha, true_fgr), true_bgr) in enumerate(tqdm(train_loader)):

                #####################################
                # train matte generator
                #####################################
                set_requires_grad([self.discrimintaor], False)
                g_optimizer.zero_grad()

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
                    g_loss = compute_refine_loss(
                        pred_pha,
                        pred_fgr,
                        pred_pha_sm,
                        pred_fgr_sm,
                        pred_err_sm,
                        true_pha,
                        true_fgr,
                    )

                    # Judge by discriminator
                    # Composite foreground onto source (predicted)
                    pred_src = pred_fgr * pred_pha + true_bgr * (1 - pred_pha)
                    pred_d = torch.cat([pred_src, pred_pha], dim=1)
                    true_d = torch.cat([true_src, true_pha], dim=1)
                    input_disc = kornia.resize(
                        torch.cat([true_d, pred_d.detach()], dim=0),
                        pred_pha_sm.shape[2:],
                    )

                    disc_judge = self.discrimintaor(input_disc.clone())
                    judge_true, judge_pred = torch.split(
                        disc_judge, dim=0, split_size_or_sections=2
                    )

                    g_loss_d = gan_loss(judge_pred, judge_true)
                    loss_g = g_loss + self.gan_weight * g_loss_d
                scaler.scale(loss_g).backward()
                scaler.step(g_optimizer)
                scaler.update()

                #########################################
                # train the discriminator
                #########################################
                set_requires_grad([self.discrimintaor], True)
                d_optimizer.zero_grad()
                with autocast(enabled=torch.cuda.is_available()):
                    disc_judge = self.discrimintaor(input_disc)
                    judge_true, judge_pred = torch.split(
                        disc_judge, dim=0, split_size_or_sections=2
                    )
                    loss_d = gan_loss(judge_true, judge_pred)
                scaler.scale(loss_d).backward()
                scaler.step(d_optimizer)
                scaler.update()

                if (i + 1) % int(10 * 2 / batch_size) == 0:
                    writer.add_scalar("loss", g_loss, step)

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

            current_val_loss = valid(self.model, "refine", valid_loader, writer, step)

            if current_val_loss < valid_loss:
                valid_loss = current_val_loss
                torch.save(
                    self.model.state_dict(),
                    f"checkpoint/matting_gan/{now}/epoch-{epoch}-loss-{valid_loss:.4f}.pth",
                )
