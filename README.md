# COCOLA
## Introduction
TODO

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

## Usage
This project uses [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html).
For info about usage:
```
python main.py --help
```
### Example: Training a contrastive model
```
python main.py fit --config configs/train_all_submixtures_efficientnet.yaml
```

## Troubleshooting
### CocoChorales Dataset
Remove `string_track001353` from the `train` split as one stem contains less frames than the other ones.