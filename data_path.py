"""
Module for recording directory paths to different datasets.
All datasets follows the following format, where fgr and pha points to directory that contains jpg or png.
Inside the directory could be any nested formats, but fgr and pha structure must match. You can add your own
dataset to the list as long as it follows the format. 'fgr' should point to foreground images with RGB channels,
'pha' should point to alpha images with only 1 grey channel.
{
    'YOUR_DATASET': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        }
    }
}
"""

DATA_PATH = {
    "PhotoMatte85": {
        "train": {
            "fgr": "datasets/PhotoMatte85/train",
            "pha": "datasets/PhotoMatte85/train",
        }
    },
    "alphamatting": {
        "valid": {
            "fgr": "datasets/alphamatting/valid",
            "pha": "datasets/alphamatting/valid",
        }
    },
    "backgrounds": {
        "train": "datasets/backgrounds/train",
        "valid": "datasets/backgrounds/valid",
    },
}
