"""
The CoCOLA Contrastive Model.
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

from contrastive_model import constants


class BilinearSimilarity(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(data=torch.Tensor(self.dim, self.dim))
        self.w.data.normal_(0, 0.05)

    def forward(self, x, y):
        projection_x = torch.matmul(self.w, y.t())
        similarities = torch.matmul(x, projection_x)
        return similarities


class EfficientNetEncoder(nn.Module):
    def __init__(self,
                 dropout_p,
                 rand_mask,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.input_type = input_type
        self.rand_mask = rand_mask

        in_channels = 2 if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE else 1
        self.model = nn.Sequential(
            EfficientNet.from_name(
                "efficientnet-b0", include_top=False, in_channels=in_channels),
            nn.Dropout(self.dropout_p),
            nn.Flatten()
        )

    def forward(self, x):
        if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE:
            # One of the three masks is applied to each element of the batch with same probability:
            # 1. The first channel is set to 0s
            # 2. The second channel is set to 0s
            # 3. None of the channels is set to 0s
            if self.rand_mask:
                choices = torch.randint(0, 3, (x.shape[0],))
                x[choices == 0, 0, :, :] = 0
                x[choices == 1, 1, :, :] = 0
            embeddings = self.model(x)
        elif self.input_type == constants.ModelInputType.SINGLE_CHANNEL_HARMONIC:
            embeddings = self.model(x[:, 0, :, :].unsqueeze(1))
        elif self.input_type == constants.ModelInputType.SINGLE_CHANNEL_PERCUSSIVE:
            embeddings = self.model(x[:, 1, :, :].unsqueeze(1))
        return embeddings


class CoColaEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int = 512,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 rand_mask: bool = False,
                 dropout_p: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_type = input_type
        self.dropout_p = dropout_p
        self.rand_mask = rand_mask

        self.encoder = EfficientNetEncoder(
            dropout_p=self.dropout_p, input_type=self.input_type, rand_mask=self.rand_mask)
        self.projection = nn.Linear(1280, self.embedding_dim)

    def forward(self, x):
        anchor, positive = x["anchor"], x["positive"]

        data = torch.cat((anchor, positive), dim=0)

        embeddings = self.encoder(data)
        projected = self.projection(embeddings)

        anchor_projected, positive_projected = torch.split(
            projected, projected.size(0) // 2, dim=0)

        return anchor_projected, positive_projected


class CoCola(L.LightningModule):
    def __init__(self,
                 learning_rate: float = 0.001,
                 embedding_dim: int = 512,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 rand_mask: bool = False,
                 dropout_p: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.input_type = input_type
        self.dropout_p = dropout_p
        self.rand_mask = rand_mask

        self.encoder = CoColaEncoder(embedding_dim=self.embedding_dim,
                                     input_type=self.input_type,
                                     dropout_p=self.dropout_p, rand_mask=self.rand_mask)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh()
        self.similarity = BilinearSimilarity(dim=self.embedding_dim)

    def forward(self, x):
        anchor_embedding, positive_embedding = self.encoder(x)
        anchor_embedding = self.tanh(self.layer_norm(anchor_embedding))
        positive_embedding = self.tanh(self.layer_norm(positive_embedding))

        similarities = self.similarity(anchor_embedding, positive_embedding)
        return similarities

    def training_step(self, x, batch_idx):
        similarities = self(x)
        sparse_labels = torch.arange(
            similarities.size(0), device=similarities.device)

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, x, batch_idx):
        similarities = self(x)
        sparse_labels = torch.arange(
            similarities.size(0), device=similarities.device)

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

    def test_step(self, x, batch_idx):
        similarities = self(x)
        sparse_labels = torch.arange(
            similarities.size(0), device=similarities.device)

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
