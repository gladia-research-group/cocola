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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the matrix of similarities between the elements of x and y.

        Args:
            x (torch.Tensor): the first batch of embeddings of shape (B, E).
            y (torch.Tensor): the second batch of embeddings of shape (B, E). 

        Returns:
            torch.Tensor: the matrix of similarities of shape (B, B).
        """
        projection_y = torch.matmul(self.w, y.t())
        similarities = torch.matmul(x, projection_y)
        return similarities
    
    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the vector of pairwise similarities between the elements of x and y.

        Args:
            x (torch.Tensor): the first batch of embeddings of shape (B, E).
            y (torch.Tensor): the second batch of embeddings of shape (B, E). 

        Returns:
            torch.Tensor: the vector of similarities of shape (B).
        """
        projection_y = torch.matmul(self.w, y.t())
        similarities = (x * projection_y.t()).sum(dim=-1)
        return similarities


class EfficientNetEncoder(nn.Module):
    def __init__(self,
                 dropout_p,
                 in_channels: int = 2) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.in_channels = in_channels

        self.model = nn.Sequential(
            EfficientNet.from_name(
                "efficientnet-b0", include_top=False, in_channels=self.in_channels),
            nn.Dropout(self.dropout_p),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CoColaEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int = 512,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 dropout_p: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_type = input_type
        self.dropout_p = dropout_p

        in_channels = 2 if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE else 1
        self.encoder = EfficientNetEncoder(
            dropout_p=self.dropout_p, in_channels=in_channels)
        self.projection = nn.Linear(1280, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        projected = self.projection(embeddings)
        return projected


class CoCola(L.LightningModule):
    def __init__(self,
                 learning_rate: float = 0.001,
                 embedding_dim: int = 512,
                 embedding_mode: constants.EmbeddingMode = constants.EmbeddingMode.RANDOM,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 dropout_p: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.embedding_mode = embedding_mode
        self.input_type = input_type
        self.dropout_p = dropout_p
        
        self.encoder = CoColaEncoder(embedding_dim=self.embedding_dim,
                                     input_type=self.input_type,
                                     dropout_p=self.dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh()
        self.similarity = BilinearSimilarity(dim=self.embedding_dim)

    def set_embedding_mode(self, embedding_mode: constants.EmbeddingMode):
        """Sets the embedding mode for inference (for DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE models only).

        The embedding mode specifies the channel(s) to be used at inference time, a 0-mask is applied
        on the other channel(s):
        - HARMONIC: a 0-mask is applied on the percussive channel
        - PERCUSSIVE: a 0-mask is applied on the harmonic channel
        - BOTH: keeps both channels
        - RANDOM: applies one of the previous three transformations at random to each element of a batch.

        Args:
            embedding_mode (constants.EmbeddingMode): the embedding mode.
        """
        self.embedding_mode = embedding_mode
        self.encoder.embedding_mode = embedding_mode

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the COCOLA score between each element of x and y. 

        Args:
            x (torch.Tensor): the first batch of spectrograms of shape (B, C, H, W).
            y (torch.Tensor): the second batch of spectrograms of shape (B, C, H, W).

        Returns:
            torch.Tensor: the batch of COCOLA scores of shape (B).
        """
        data = torch.cat((x, y), dim=0)
        data_embeddings = self.encoder(data)
        x_embeddings, y_embeddings = torch.split(
            data_embeddings, data_embeddings.size(0) // 2, dim=0)

        scores = self.similarity.pairwise(x_embeddings, y_embeddings)
        return scores

    def forward(self, x, use_anchors_as_negatives=True):
        anchors, positives = x["anchor"], x["positive"]
        if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE:
            if self.embedding_mode == constants.EmbeddingMode.RANDOM:
                choices = torch.randint(0, 3, (anchors.shape[0],))
                anchors[choices == 0, 0, :, :] = 0
                anchors[choices == 1, 1, :, :] = 0

                positives[choices == 0, 0, :, :] = 0
                positives[choices == 1, 1, :, :] = 0
            elif self.embedding_mode == constants.EmbeddingMode.HARMONIC:
                anchors[:, 1, :, :] = 0
                positives[:, 1, :, :] = 0
            elif self.embedding_mode == constants.EmbeddingMode.PERCUSSIVE:
                anchors[:, 0, :, :] = 0
                positives[:, 0, :, :] = 0

        data = torch.cat((anchors, positives), dim=0)
        data_embeddings = self.encoder(data)
        anchor_embeddings, positive_embeddings = torch.split(
            data_embeddings, data_embeddings.size(0) // 2, dim=0)
        
        anchor_embeddings = self.tanh(self.layer_norm(anchor_embeddings))
        positive_embeddings = self.tanh(self.layer_norm(positive_embeddings))
        if use_anchors_as_negatives:
            embeddings = torch.cat((anchor_embeddings, positive_embeddings), 0)

            similarities = self.similarity(embeddings, embeddings)
            batch_size = anchor_embeddings.shape[0]
            mask = torch.eye(
                2 * batch_size, dtype=torch.bool, device=similarities.device)
            similarities = similarities[~mask].reshape(2 * batch_size, -1)
        else:
            similarities = self.similarity(anchor_embeddings, positive_embeddings)
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
        similarities = self(x, use_anchors_as_negatives=False)
        sparse_labels = torch.arange(
            similarities.size(0), device=similarities.device)

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
