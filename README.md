# COCOLA
## Introduction
This is the official repository [COCOLA: Coherence-Oriented Contrastive Learning of Musical Audio Representations](https://arxiv.org/abs/2404.16969).

![alt text](res/logo.jpg)


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
See `configs` for examples of cofig files.

### Example: Training a contrastive model on CocoChorales + MoisesDB + Slakh2100
```
python main.py fit --config configs/train_all_submixtures.yaml
```

## Pretrained Models
| Model Checkpoint | Train Dataset | Train Config | Description |
|-------|---------|-------------|---------|
| https://drive.google.com/file/d/1HdKgDV2wCdGwCWPlIIRm2ytlbUNah8fo/view?usp=sharing  | Moisesdb, Slakh2100, CocoChorales| batch size = 32, chunk duration = 5s, embedding mode = EmbeddingMode.RANDOM, input type = ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE | TODO    |

### Example 1: calculating COCOLA Scores on a batch of pairs of music audio.
```python
from contrastive_model.contrastive_model import CoCola

model = CoCola.load_from_checkpoint("/path/to/checkpoint.ckpt")

model.eval()

scores = model.score(x, y)
```
where `x` and `y` are tensors of shape `[B, 1, 16000*5]` (`B` audio tracks of 5 seconds sampled at 16000 kHz).

`scores[i]` contains the COCOLA score between `x[i]` and `y[i]`.

### Example 2: calculating COCOLA cross-scores matrix on a batch of pairs of music audio.
```python
from contrastive_model.contrastive_model import CoCola

model = CoCola.load_from_checkpoint("/path/to/checkpoint.ckpt")

model.eval()

scores = model(x)
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
