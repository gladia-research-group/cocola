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

## Train the contrastive model
This project uses [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html).
For info about usage:
```
python main.py --help
```

## Troubleshooting
### CocoChorales Dataset
Remove `string_track001353` from the `train` split as one stem contains less frames than the other ones.