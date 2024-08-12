from lightning.pytorch.cli import LightningCLI
import torch
torch.set_float32_matmul_precision('high')

from contrastive_model.contrastive_model_data import CoColaDataModule
from contrastive_model.contrastive_model import CoCola


def cli_main():
    cli = LightningCLI(CoCola, CoColaDataModule) #save_config_callback=None


if __name__ == "__main__":
    cli_main()
