# COCOLA
## Introduction
This is the official repository [COCOLA: Coherence-Oriented Contrastive Learning of Musical Audio Representations](https://arxiv.org/abs/2404.16969).

COCOLA is a contrastive model which is able to estimate the harmonic and rythmic coherence between pairs of music audio examples.

![alt text](assets/logo.jpg)


## Installation
### Create virtual environment (Optional)
```
conda create --name cocola python=3.11
conda activate cocola
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install datasets
If you wish to use MoisesDB for training/validation/test, download it from the [official website](https://music.ai/research/) and unzip it inside `~/moisesdb_contrastive`.
The other datasets ([CocoChorales](https://magenta.tensorflow.org/datasets/cocochorales), [Slakh2100](http://www.slakh.com), [Musdb](https://sigsep.github.io/datasets/musdb.html)) are automatically downoladed and extracted by the respective PyTorch Datasets.

## Usage
This project uses [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html).
For info about usage:
```
python main.py --help
```
For info about subcommands usage:
```
python main.py fit --help
python main.py validate --help
python main.py test --help
python main.py predict --help
```
You can pass a YAML config file as command line argument instead of specifying each parameter in the command:
```
python main.py fit --config path/to/config.yaml
```
See `configs` for examples of config files.

### Example: Training a contrastive model on CocoChorales + MoisesDB + Slakh2100
```
python main.py fit --config configs/train_all_submixtures.yaml
```

## Pretrained Models
Model Name | Model Checkpoint | Train Dataset | Train Config File | Description |
|-------|-------|---------|-------------|---------|
COCOLA_HP_v1| https://drive.google.com/file/d/1HdKgDV2wCdGwCWPlIIRm2ytlbUNah8fo/view?usp=sharing  | Moisesdb, Slakh2100, CocoChorales| `configs/train_all_submixtures_hpss.yaml`| Allows to compute COCOLA Score, COCOLA Harmonic Score and COCOLA Percussive Score.|

### Example 1: calculating COCOLA (Harmonic/Percussive) Score with COCOLA_HP_v1 on a of pair of music audio examples.
```python
from contrastive_model import constants
from contrastive_model.contrastive_model import CoCola
from feature_extraction.feature_extraction import CoColaFeatureExtractor

model = CoCola.load_from_checkpoint("/path/to/checkpoint.ckpt")
feature_extractor = CocolaFeatureExtractor()

model.eval()

# Set to:
# - constants.EmbeddingMode.BOTH for standard COCOLA Score
# - constants.EmbeddingMode.HARMONIC for COCOLA Harmonic Score
# - constants.EmbeddingMode.PERCUSSIVE for COCOLA Percussive Score
model.set_embedding_mode(constants.EmbeddingMode.BOTH)

features_x = feature_extractor(x)
features_y = feature_extractor(y)
score = model.score(features_x, features_y)
```
where `x` and `y` are tensors of shape `[1, 16000*5]` (audio tracks of 5 seconds sampled at 16000 kHz).

### Example 2: calculating COCOLA (Harmonic/Percussive) Scores with COCOLA_HP_v1 on a batch of pairs of music audio examples.
```python
from contrastive_model import constants
from contrastive_model.contrastive_model import CoCola
from feature_extraction.feature_extraction import CoColaFeatureExtractor

model = CoCola.load_from_checkpoint("/path/to/checkpoint.ckpt")
feature_extractor = CocolaFeatureExtractor()

model.eval()

# Set to:
# - constants.EmbeddingMode.BOTH for standard COCOLA Score
# - constants.EmbeddingMode.HARMONIC for COCOLA Harmonic Score
# - constants.EmbeddingMode.PERCUSSIVE for COCOLA Percussive Score
model.set_embedding_mode(constants.EmbeddingMode.BOTH)

features_x = feature_extractor(x)
features_y = feature_extractor(y)
scores = model.score(x, y)
```
where `x` and `y` are tensors of shape `[B, 1, 16000*5]` (`B` audio tracks of 5 seconds sampled at 16000 kHz).

`scores[i]` contains the COCOLA score between `x[i]` and `y[i]`.

### Example 3: calculating COCOLA (Harmonic/Percussive) cross-scores matrix with COCOLA_HP_v1 on a batch of pairs of music audio examples.
```python
from contrastive_model import constants
from contrastive_model.contrastive_model import CoCola
from feature_extraction.feature_extraction import CoColaFeatureExtractor

model = CoCola.load_from_checkpoint("/path/to/checkpoint.ckpt")
feature_extractor = CocolaFeatureExtractor()

model.eval()

# Set to:
# - constants.EmbeddingMode.BOTH for standard COCOLA Score
# - constants.EmbeddingMode.HARMONIC for COCOLA Harmonic Score
# - constants.EmbeddingMode.PERCUSSIVE for COCOLA Percussive Score
model.set_embedding_mode(constants.EmbeddingMode.BOTH)

features = feature_extractor(x)
scores = model(features)
```
where `x` is like:
```python
x = {
    "anchor": torch.randn(batch_size, 1, 16000*5, dtype=torch.float32), # 5 seconds, 16000 kHz
    "positive": torch.randn(batch_size, 1, 16000*5, dtype=torch.float32) # 5 seconds, 16000 kHz
}
```
`scores[i, j]` contains the COCOLA score between `x['anchor'][i]` and `y['positive'][j]`.

## Troubleshooting
### CocoChorales Dataset
Remove `string_track001353` from the `train` split as one stem contains less frames than the other ones.
