from lightning.pytorch.cli import LightningCLI
import torch

from contrastive_model.contrastive_model_data import CoColaDataModule
from contrastive_model.contrastive_model import CoCola

import logging

torch.set_float32_matmul_precision('high')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def cli_main():
    cli = LightningCLI(CoCola, CoColaDataModule, save_config_callback=None)


if __name__ == "__main__":
    cli_main()
