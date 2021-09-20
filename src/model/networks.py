"""module for auxillary networks"""
import functools

import torch
from torch import nn


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    """https://github.com/CDOTAD/AlphaGAN-Matting/blob/fa0f4ee3515ed49a10faf10e252d29e8055b8769/
    model/NLayerDiscriminator.py#L7
    """

    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False
    ):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel, stride=2, padding=padw),
            nn.LeakyReLU(0.2),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for layer_index in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** layer_index, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kernel,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kernel,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """forward pass for the neural network"""
        return self.model(inputs)


if __name__ == "__main__":
    Disc = NLayerDiscriminator(input_nc=4, n_layers=4, norm_layer=nn.BatchNorm2d)
