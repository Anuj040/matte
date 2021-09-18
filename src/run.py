"""main controller module"""
import argparse
import sys

import torch

sys.path.append(("./"))

# pylint: disable = wrong-import-position, import-error
from src.train_base import CoarseMatte
from src.train_refine import FineMatte


def common_parser() -> argparse.Namespace:
    """Prepares a list of common cl args for all methods

    Returns:
        argparse.Namespace: command line args
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train"])
    parser.add_argument(
        "--model_type", default="base", type=str, choices=["base", "refine"]
    )
    parser.add_argument("--load_base", default=False, type=bool)

    return parser.parse_args()


def main() -> None:
    """main control method"""

    args = common_parser()
    if args.model_type == "base":
        matte = CoarseMatte()
    elif args.model_type == "refine":
        matte = FineMatte()

        if args.load_base:
            # get the pretrained weights
            model_path = (
                "checkpoint/matting_base/18092021_114312/"
                "epoch-8-loss-0.06849407218396664.pth"
            )
            pretrained_dict = torch.load(model_path)
            model_dict = matte.model.state_dict()
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            matte.model.load_state_dict(model_dict)

    if args.mode == "train":
        matte.train()


if __name__ == "__main__":
    main()
