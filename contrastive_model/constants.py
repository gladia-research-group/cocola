"""Look up for constants."""

import enum


@enum.unique
class Dataset(enum.Enum):
    """Look up for dataset names."""

    CCS = "coco_chorales_contrastive/*"

    CCS_RANDOM = "coco_chorales_contrastive/random"

    CCS_STRING = "coco_chorales_contrastive/string"

    CCS_BRASS = "coco_chorales_contrastive/brass"

    CCS_WOODWIND = "coco_chorales_contrastive/woodwind"

    SLAKH2100 = "slakh2100_contrastive"

    MOISESDB = "moisesdb_contrastive"

    MIXED = "mixed_contrastive"


class ModelInputType(enum.Enum):
    """Look up for CoCola HPSS Model input types."""

    SINGLE_CHANNEL_HARMONIC = 'single_channel_harmonic'

    SINGLE_CHANNEL_PERCUSSIVE = 'single_channel_percussive'

    DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE = 'double_channel_harmonic_percussive'


class Logger(enum.Enum):
    """Look up for loggers"""

    WANDB = "wandb"

    TENSORBOARD = "tensorboard"
