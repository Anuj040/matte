"""module for training refinement model"""
import torch

from model import MattingRefine

# check if cuda available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FineMatte:
    """class for building, training refined matte generator model"""

    def __init__(self) -> None:
        self.model = MattingRefine("resnet50").to(DEVICE)


if __name__ == "__main__":
    matte = FineMatte()
