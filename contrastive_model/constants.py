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

class EmbeddingModel(enum.Enum):
    """Look up for embedding models."""

    EFFICIENTNET = "efficientnet"

    CLAP = "clap"

class Similarity(enum.Enum):
    """Look up for similarity types."""

    BILINEAR = "bilinear"

    COSINE = "dot"


class Logger(enum.Enum):
    """Look up for loggers"""

    WANDB = "wandb"

    TENSORBOARD = "tensorboard"
