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


@enum.unique
class ModelInputType(enum.Enum):
    """Look up for CoCola Model input types."""
    SINGLE_CHANNEL_MEL_SPECTROGRAM = 'single_channel_mel_spectrogram'

    DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE = 'double_channel_harmonic_percussive'


@enum.unique
class ModelFeatureExtractorType(enum.Enum):
    """Look up for CoCola data feature extraction types."""
    HPSS = 'hpss'

    MEL_SPECTROGRAM = 'mel_spectrogram'


@enum.unique
class FeatureExtractionTime(enum.Enum):
    """Look up for CoCola data feature extraction time."""
    OFFLINE = 'offline'

    ONLINE = 'online'


@enum.unique
class EmbeddingMode(enum.Enum):
    """Look up for selecting channels to use at inference time for DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE models."""
    HARMONIC = 'harmonic'

    PERCUSSIVE = 'percussive'

    BOTH = 'both'

    RANDOM = 'random'
