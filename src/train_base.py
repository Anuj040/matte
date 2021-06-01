"""module for training model base"""
from model import MattingBase


class CoarseMatte:
    """class for building, training coarse matte generator model"""

    def __init__(self) -> None:
        self.model = MattingBase("resnet50")


if __name__ == "__main__":
    matte = CoarseMatte()
