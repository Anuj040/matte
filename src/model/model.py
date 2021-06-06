"""module defining model architectures"""
import torch
from torch import nn
from torchvision.models.segmentation.deeplabv3 import ASPP

from .decoder import Decoder
from .resnet import ResNetEncoder


class Base(nn.Module):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """

    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ["resnet50", "resnet101"]:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder(
                [256, 128, 64, 48, out_channels], [512, 256, 64, in_channels]
            )
        else:
            raise NotImplementedError(f"{backbone} backbone has not been implemented")

    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        return x


class MattingBase(Base):
    """
    MattingBase is used to produce coarse global results at a lower resolution.
    MattingBase extends Base.

    Args:
        backbone: ["resnet50", "resnet101", "mobilenetv2"]

    Input:
        src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        bgr: (B, 3, H, W) the background image . Channels are RGB values normalized to 0 ~ 1.

    Output:
        pha: (B, 1, H, W) the alpha prediction. Normalized to 0 ~ 1.
        fgr: (B, 3, H, W) the foreground prediction. Channels are RGB values normalized to 0 ~ 1.
        err: (B, 1, H, W) the error prediction. Normalized to 0 ~ 1.
        hid: (B, 32, H, W) the hidden encoding. Used for connecting refiner module.

    Example:
        model = MattingBase(backbone='resnet50')

        pha, fgr, err, hid = model(src, bgr)    # for training
        pha, fgr = model(src, bgr)[:2]          # for inference
    """

    def __init__(self, backbone: str):
        super().__init__(backbone, in_channels=6, out_channels=(1 + 3 + 1 + 32))

    def forward(self, src, bgr):
        x = torch.cat([src, bgr], dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha = x[:, 0:1].clamp_(0.0, 1.0)
        fgr = x[:, 1:4].add(src).clamp_(0.0, 1.0)
        err = x[:, 4:5].clamp_(0.0, 1.0)
        hid = x[:, 5:].relu_()
        return pha, fgr, err, hid
