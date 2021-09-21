"""utility functions for model training"""

import kornia
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_base_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr):
    """loss calculating function"""
    true_err = torch.abs(pred_pha.detach() - true_pha)
    true_msk = true_pha != 0
    return (
        F.l1_loss(pred_pha, true_pha)
        + F.l1_loss(kornia.sobel(pred_pha), kornia.sobel(true_pha))
        + F.l1_loss(pred_fgr * true_msk, true_fgr * true_msk)
        + F.mse_loss(pred_err, true_err)
    )


def compute_refine_loss(
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


# pylint: disable = too-many-locals
def valid(
    model: nn.Module,
    model_type: str,
    dataloader: DataLoader,
    writer: SummaryWriter,
    step: int,
) -> float:
    """model evaluation step executor

    Args:
        model_type (str, opt): refined or coarse alpha matte
        model (nn.Module): [description]
        dataloader (DataLoader): [description]
        writer (SummaryWriter): [description]
        step (int): [description]

    Returns:
        float: validation loss
    """
    assert model_type in ["base", "refine"]

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
            if model_type == "base":
                pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
                loss = compute_base_loss(
                    pred_pha, pred_fgr, pred_err, true_pha, true_fgr
                )
                loss_total += loss.cpu().item() * batch_size
                loss_count += batch_size
            elif model_type == "refine":
                pred_pha, pred_fgr, pred_pha_sm, pred_fgr_sm, pred_err_sm, _ = model(
                    true_src, true_bgr
                )
                loss = compute_refine_loss(
                    pred_pha,
                    pred_fgr,
                    pred_pha_sm,
                    pred_fgr_sm,
                    pred_err_sm,
                    true_pha,
                    true_fgr,
                )

    writer.add_scalar("valid_loss", loss_total / loss_count, step)
    model.train()
    return loss_total / loss_count


def set_requires_grad(nets, requires_grad=False):
    """[summary]

    Args:
        nets ([type]): [description]
        requires_grad (bool, optional): [description]. Defaults to False.
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def gan_loss(judge_1: torch.Tensor, judge_2: torch.Tensor) -> torch.Tensor:
    """loss calculating function"""
    loss = torch.mean(
        judge_1 - torch.mean(judge_2, dim=(0, 2, 3)) - 1.0, dim=0
    ) + torch.mean(judge_2 - torch.mean(judge_1, dim=(0, 2, 3)) + 1.0, dim=0)
    return torch.mean(loss)
