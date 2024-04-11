"""
The CoCOLA Contrastive Model.
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio.transforms as T
from efficientnet_pytorch import EfficientNet
from transformers import ClapModel, ClapFeatureExtractor

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


class ClapEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = ClapFeatureExtractor.from_pretrained(
            "laion/clap-htsat-unfused", sampling_rate=16000)
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.model.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            device = x.device
            x = x.squeeze(1).cpu().numpy()
            processed_x = self.processor(
                raw_speech=x, return_tensors="pt", sampling_rate=16000).to(device)
        embeddings = self.model.get_audio_features(**processed_x)
        return embeddings


class EfficientNetEncoder(nn.Module):
    def __init__(self, dropout_p) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.processor = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                win_length=400,
                hop_length=160,
                f_min=60.0,
                f_max=7800.0,
                n_mels=64,
            ),
            T.AmplitudeToDB()
        )
        self.model = nn.Sequential(
            EfficientNet.from_name(
                "efficientnet-b0", include_top=False, in_channels=1),
            nn.Dropout(self.dropout_p),
            nn.Flatten()
        )

        self.processor.requires_grad_(False)

    def forward(self, x):
        processed_x = self.processor(x)
        embeddings = self.model(processed_x)
        return embeddings


class CoColaEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int = 512,
                 embedding_model: constants.EmbeddingModel = constants.EmbeddingModel.EFFICIENTNET,
                 dropout_p: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        if self.embedding_model == constants.EmbeddingModel.EFFICIENTNET:
            self.dropout_p = dropout_p
            self.encoder = EfficientNetEncoder(dropout_p=self.dropout_p)
            self.projection = nn.Linear(1280, self.embedding_dim)

        elif self.embedding_model == constants.EmbeddingModel.CLAP:
            self.encoder = ClapEncoder()
            self.projection = nn.Linear(512, self.embedding_dim)

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
                 embedding_model: constants.EmbeddingModel = constants.EmbeddingModel.EFFICIENTNET,
                 dropout_p: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.dropout_p = dropout_p

        self.encoder = CoColaEncoder(embedding_dim=self.embedding_dim,
                                     embedding_model=self.embedding_model,
                                     dropout_p=self.dropout_p)
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
