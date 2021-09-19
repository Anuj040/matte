"""
Pair transforms are MODs of regular transforms so that it takes in multiple images
and apply exact transforms on all images. This is especially useful when we want the
transforms on a pair of images.
Example:
    img1, img2, ..., imgN = transforms(img1, img2, ..., imgN)
"""
# pylint: disable = R0903: too-few-public-methods
import math
import random
from typing import Tuple

import kornia
import torch
from PIL import ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as F


class PairCompose(T.Compose):
    """Extends the T.compose class to a pair of images

    Args:
        T.Compose: Parent class
    """

    def __call__(self, *x):
        """calls the transform on the pair"""
        for transform in self.transforms:
            x = transform(*x)
        return x


class PairApply:
    """Applies a transform on a pair of images"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *x) -> list:
        """calls the transform on the pair"""
        return [self.transforms(xi) for xi in x]


class PairApplyOnlyAtIndices:
    """method to apply a transformation only to elements from given indices"""

    def __init__(self, indices: list, transforms):
        self.indices = indices
        self.transforms = transforms

    def __call__(self, *x) -> list:
        return [
            self.transforms(xi) if i in self.indices else xi for i, xi in enumerate(x)
        ]


class PairRandomAffine(T.RandomAffine):
    """Extends the T.RandomAffine class to apply same transformation on a pair of images"""

    # pylint: disable = R0913 #too-many-arguments
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        resamples=None,
        fillcolor=0,
    ):
        super().__init__(
            degrees, translate, scale, shear, T.InterpolationMode.NEAREST, fillcolor
        )
        self.resamples = resamples

    def __call__(self, *x):
        if len(x) == 0:
            return []

        param = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, x[0].size
        )
        resamples = self.resamples or [self.resample] * len(x)
        return [
            F.affine(xi, *param, resamples[i], self.fillcolor) for i, xi in enumerate(x)
        ]


class PairRandomHorizontalFlip(T.RandomHorizontalFlip):
    """Extends T.RandomHorizontalFlip to apply simultaneous horizontal flip on a pair of images"""

    def __call__(self, *x):
        if torch.rand(1) < self.p:
            x = [F.hflip(xi) for xi in x]
        return x


class RandomBoxBlur:
    """Applies Box blur on an image"""

    def __init__(self, prob, max_radius):
        self.prob = prob
        self.max_radius = max_radius

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            img = img.filter(fil)
        return img


class PairRandomBoxBlur(RandomBoxBlur):
    """Extends RandomBoxBlur class to apply same box blur on a pair of images"""

    def __call__(self, *x):
        if torch.rand(1) < self.prob:
            fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            x = [xi.filter(fil) for xi in x]
        return x


class RandomSharpen:
    """Applies sharpening transformation on an image"""

    def __init__(self, prob):
        self.prob = prob
        self.filter = ImageFilter.SHARPEN

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = img.filter(self.filter)
        return img


class PairRandomSharpen(RandomSharpen):
    """Extends RandomSharpen class to apply same sharpening trasnform to a pair of images"""

    def __call__(self, *x):
        if torch.rand(1) < self.prob:
            x = [xi.filter(self.filter) for xi in x]
        return x


class PairRandomAffineAndResize:
    """Applies same random Affine and resize tranform to pair of images"""

    # pylint: disable = R0913 # too-many-arguments
    def __init__(
        self,
        size,
        degrees,
        translate,
        scale,
        shear,
        resample=T.InterpolationMode.BILINEAR,
        fillcolor=0,
    ):
        self.size = size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, *x):
        if len(x) == 0:
            return []

        width, height = x[0].size
        scale_factor = max(self.size[1] / width, self.size[0] / height)

        w_padded = max(width, self.size[1])
        h_padded = max(height, self.size[0])

        pad_h = int(math.ceil((h_padded - height) / 2))
        pad_w = int(math.ceil((w_padded - width) / 2))

        scale = self.scale[0] * scale_factor, self.scale[1] * scale_factor
        translate = self.translate[0] * scale_factor, self.translate[1] * scale_factor
        affine_params = T.RandomAffine.get_params(
            self.degrees, translate, scale, self.shear, (width, height)
        )

        def transform(img):
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (pad_w, pad_h))

            img = F.affine(img, *affine_params, self.resample, self.fillcolor)
            img = F.center_crop(img, self.size)
            return img

        return [transform(xi) for xi in x]


# pylint: disable = W0221: arguments-differ
class RandomAffineAndResize(PairRandomAffineAndResize):
    """Applies random Affine and resize tranform to a single image"""

    def __call__(self, img):
        return super().__call__(img)[0]


def train_step_augmenter(src: torch.Tensor, bgr: torch.Tensor) -> Tuple[torch.Tensor]:
    """train step augmentations for source (composite) and background image

    Args:
        src (torch.Tensor): source (composite) image
        bgr (torch.Tensor): background image

    Returns:
        Tuple[torch.Tensor]: augmented source, bgr images
    """

    # Augment with noise
    aug_noise_idx = torch.rand(len(src)) < 0.4
    if aug_noise_idx.any():
        src[aug_noise_idx] = (
            src[aug_noise_idx]
            .add_(torch.randn_like(src[aug_noise_idx]).mul_(0.03 * random.random()))
            .clamp_(0, 1)
        )
        bgr[aug_noise_idx] = (
            bgr[aug_noise_idx]
            .add_(torch.randn_like(bgr[aug_noise_idx]).mul_(0.03 * random.random()))
            .clamp_(0, 1)
        )
    del aug_noise_idx

    # Augment background with jitter
    aug_jitter_idx = torch.rand(len(bgr)) < 0.8
    if aug_jitter_idx.any():
        bgr[aug_jitter_idx] = kornia.augmentation.ColorJitter(0.18, 0.18, 0.18, 0.1)(
            bgr[aug_jitter_idx]
        )
    del aug_jitter_idx

    # Augment background with affine
    aug_affine_idx = torch.rand(len(bgr)) < 0.3
    if aug_affine_idx.any():
        bgr[aug_affine_idx] = T.RandomAffine(degrees=(-1, 1), translate=(0.01, 0.01))(
            bgr[aug_affine_idx]
        )
    del aug_affine_idx

    return src, bgr
