# COCOLA
## Introduction
This is the official repository [COCOLA: Coherence-Oriented Contrastive Learning of Musical Audio Representations](https://arxiv.org/abs/2404.16969)

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
See `config` for examples of cofig files.

### Example: Training a contrastive model on CocoChorales + MoisesDB + Slakh2100
```
python main.py fit --config configs/train_all_submixtures_efficientnet.yaml
```

## Troubleshooting
### CocoChorales Dataset
Remove `string_track001353` from the `train` split as one stem contains less frames than the other ones.
