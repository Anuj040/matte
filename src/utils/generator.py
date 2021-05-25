"""data generator module"""
import glob
import os
import sys
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.append("./")
# pylint: disable = E0401, C0413
import src.utils.augmentation as A
from data_path import DATA_PATH


class ImagesDataset(Dataset):
    """Builds an instance of Dataset class

    Args:
        Dataset: Parent class
    """

    def __init__(
        self,
        root: str,
        mode: str = "RGB",
        shuffle: bool = False,
        transforms: Union[None, T.Compose] = None,
    ) -> None:
        """Instance intializer

        Args:
            root (str): root directory for the data
            mode (str, optional): Mode for the image object. Defaults to "RGB".
            shuffle (bool, optional): Shuffle the datset elements. Default False
            transforms (Union[None, T.Compose], optional): Image transformations.
        """
        self.shuffle = shuffle
        np.random.seed(123)
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted(
            [
                *glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True),
                *glob.glob(os.path.join(root, "**", "*.png"), recursive=True),
            ]
        )

    def __len__(self) -> int:
        """length of the dataset"""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """method to return an element of the dataset

        Args:
            idx (int): element id

        Returns:
            torch.Tensor
        """
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transforms:
            img = self.transforms(img)

        if idx + 1 == len(self.filenames) and self.shuffle:
            np.random.shuffle(self.filenames)

        return img


class ZipDataset(Dataset):
    """Dataset class to zip two datasets together

    Args:
        Dataset: Parent class
    """

    def __init__(
        self, datasets: List[Dataset], transforms=None, assert_equal_length=False
    ) -> None:
        """instance initializer

        Args:
            datasets (List[Dataset]): Dataset object(s)
            transforms ([type], optional): Transoformations for object elements. Defaults to None.
            assert_equal_length (bool, optional): check dataset size equality. Defaults to False.
        """
        super().__init__()
        self.datasets = datasets
        self.transforms = transforms

        if assert_equal_length:
            for i in range(1, len(datasets)):
                assert len(datasets[i]) == len(
                    datasets[i - 1]
                ), "Datasets are not equal in length."

    def __len__(self) -> int:
        """
        Returns:
            int: Dataset size
        """
        return max(len(d) for d in self.datasets)

    def __getitem__(self, idx: int) -> tuple:
        """method to return a set of elements from each of the datasets

        Args:
            idx (int): index for the element

        Returns:
            tuple: tuple of elements from each dataset
        """
        sample = tuple(d[idx % len(d)] for d in self.datasets)
        if self.transforms:
            sample = self.transforms(*sample)
        return sample


class DataGenerator:
    """Datagenerator class"""

    def __init__(self, dataset: str = "PhotoMatte85", mode: str = "train") -> None:
        """instance initializer

        Args:
            dataset (str, optional): Dataset to use. Defaults to "PhotoMatte85".
            mode (str, optional): Data is to be used for (train, val, etc.). Defaults to "train".
        """

        # Training DataLoader
        if mode == "train":
            _dataset = ZipDataset(
                [
                    ZipDataset(
                        [
                            # Dataset object for alpha mattes
                            ImagesDataset(DATA_PATH[dataset][mode]["pha"], mode="L"),
                            # Dataset object for foregrounds
                            ImagesDataset(DATA_PATH[dataset][mode]["fgr"], mode="RGB"),
                        ],
                        transforms=A.PairCompose(
                            [
                                A.PairRandomAffineAndResize(
                                    (512, 512),
                                    degrees=(-5, 5),
                                    translate=(0.1, 0.1),
                                    scale=(0.4, 1),
                                    shear=(-5, 5),
                                ),
                                A.PairRandomHorizontalFlip(),
                                A.PairRandomBoxBlur(0.1, 5),
                                A.PairRandomSharpen(0.1),
                                A.PairApplyOnlyAtIndices(
                                    [1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)
                                ),
                                A.PairApply(T.ToTensor()),
                            ]
                        ),
                        assert_equal_length=True,
                    ),
                    # Dataset object for backgrounds
                    ImagesDataset(
                        DATA_PATH["backgrounds"][mode],
                        mode="RGB",
                        # shuffle backgrounds with respect to foregrounds
                        shuffle=True,
                        transforms=T.Compose(
                            [
                                A.RandomAffineAndResize(
                                    (512, 512),
                                    degrees=(-5, 5),
                                    translate=(0.1, 0.1),
                                    scale=(1, 2),
                                    shear=(-5, 5),
                                ),
                                T.RandomHorizontalFlip(),
                                A.RandomBoxBlur(0.1, 5),
                                A.RandomSharpen(0.1),
                                T.ColorJitter(0.15, 0.15, 0.15, 0.05),
                                T.ToTensor(),
                            ]
                        ),
                    ),
                ]
            )
        self.dataset = _dataset

    def __call__(
        self,
        shuffle: bool = False,
        batch_size: int = 2,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Method to build the dataloader object

        Args:
            shuffle (bool, optional): Shuffle the dataset elements. Defaults to False.
            batch_size (int, optional): Defaults to 2.
            num_workers (int, optional): Defaults to 8.
            pin_memory (bool, optional): Defaults to True.

        Returns:
            DataLoader: [description]
        """
        dataloader = DataLoader(
            self.dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dataloader

    def __len__(self) -> int:
        """dataset size"""
        return len(self.dataset)


if __name__ == "__main__":
    train_set = DataGenerator()
    train_loader = train_set(shuffle=True, batch_size=2, num_workers=8, pin_memory=True)
    for item in train_loader:
        pass
        # print(item)
        # exit()
