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
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.input_type = input_type

        in_channels = 2 if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE else 1
        self.model = nn.Sequential(
            EfficientNet.from_name(
                "efficientnet-b0", include_top=False, in_channels=in_channels),
            nn.Dropout(self.dropout_p),
            nn.Flatten()
        )

    def forward(self, x):
        if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE:

            embeddings = self.model(x)
        elif self.input_type == constants.ModelInputType.SINGLE_CHANNEL_HARMONIC:
            embeddings = self.model(x[:, 0, :, :].unsqueeze(1))
        elif self.input_type == constants.ModelInputType.SINGLE_CHANNEL_PERCUSSIVE:
            embeddings = self.model(x[:, 1, :, :].unsqueeze(1))
        return embeddings


class CoColaEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int = 512,
                 rand_mask: bool = False,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 dropout_p: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_type = input_type
        self.dropout_p = dropout_p
        self.rand_mask = rand_mask
        

        self.encoder = EfficientNetEncoder(
            dropout_p=self.dropout_p, input_type=self.input_type)
        self.projection = nn.Linear(1280, self.embedding_dim)

    def forward(self, x):
        anchor, positive = x["anchor"], x["positive"]
        if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE:
            if self.rand_mask:
                choices = torch.randint(0, 3, (anchor.shape[0],))
                anchor[choices == 0, 0, :, :] = 0
                anchor[choices == 1, 1, :, :] = 0

                positive[choices == 0, 0, :, :] = 0
                positive[choices == 1, 1, :, :] = 0

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
                 rand_mask: bool = False,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 dropout_p: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.rand_mask = rand_mask
        self.input_type = input_type
        self.dropout_p = dropout_p
        
        

        self.encoder = CoColaEncoder(embedding_dim=self.embedding_dim,
                                     rand_mask=self.rand_mask,
                                     input_type=self.input_type,
                                     dropout_p=self.dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh()
        self.similarity = BilinearSimilarity(dim=self.embedding_dim)

    def forward(self, x):
        anchor_embedding, positive_embedding = self.encoder(x)
        anchor_embedding = self.tanh(self.layer_norm(anchor_embedding))
        positive_embedding = self.tanh(self.layer_norm(positive_embedding))
        embeddings = torch.cat((anchor_embedding, positive_embedding), 0)

        similarities = self.similarity(embeddings, embeddings)
        batch_size = anchor_embedding.shape[0]
        mask = torch.eye(
            2 * batch_size, dtype=torch.bool, device=similarities.device)
        similarities = similarities[~mask].reshape(2 * batch_size, -1)
        return similarities

    def training_step(self, x, batch_idx):
        similarities = self(x)
        batch_size = similarities.shape[0] // 2
        labels = torch.cat(
            (torch.arange(batch_size - 1, 2 * batch_size - 1), torch.arange(batch_size))).to(similarities.device)

        loss = F.cross_entropy(similarities, labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == labels).double().mean()

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, x, batch_idx):
        similarities = self(x)
        batch_size = similarities.shape[0] // 2
        labels = torch.cat(
            (torch.arange(batch_size - 1, 2 * batch_size - 1), torch.arange(batch_size))).to(similarities.device)

        loss = F.cross_entropy(similarities, labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == labels).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

    def test_step(self, x, batch_idx):
        similarities = self(x)
        batch_size = similarities.shape[0] // 2
        labels = torch.cat(
            (torch.arange(batch_size - 1, 2 * batch_size - 1), torch.arange(batch_size))).to(similarities.device)

        loss = F.cross_entropy(similarities, labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == labels).double().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
