import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

class Similarity(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(data=torch.Tensor(self.dim, self.dim))
        self.w.data.normal_(0, 0.05)

    def forward(self, x, y):
        projection_x = torch.matmul(self.w, y.t())
        similarities = torch.matmul(x, projection_x)
        return similarities

class CoColaEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.efficientnet_encoder = nn.Sequential(
            EfficientNet.from_name("efficientnet-b0", include_top=False, in_channels=1),
            nn.Dropout(self.dropout_p),
            nn.Flatten()
            )
        self.projection = nn.Linear(1280, self.embedding_dim)

    def forward(self, x):
        anchor, positive = x["anchor"], x["positive"]

        anchor = self.efficientnet_encoder(anchor)
        anchor = self.projection(anchor)

        positive = self.efficientnet_encoder(positive)
        positive = self.projection(positive)

        return anchor, positive

class CoCola(L.LightningModule):
    def __init__(self,
                 learning_rate: float = 0.001,
                 embedding_dim: int = 512,
                 dropout_p: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p

        self.encoder = CoColaEncoder(embedding_dim=self.embedding_dim, dropout_p=self.dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.tanh = nn.Tanh()
        self.similarity = Similarity(dim=self.embedding_dim)

    def training_step(self, x, batch_idx):
        anchor_embedding, positive_embedding = self.encoder(x)
        anchor_embedding = self.tanh(self.layer_norm(anchor_embedding))
        positive_embedding = self.tanh(self.layer_norm(positive_embedding))

        sparse_labels = torch.arange(anchor_embedding.size(0), device=anchor_embedding.device)

        similarities = self.similarity(anchor_embedding, positive_embedding)

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, x, batch_idx):
        anchor_embedding, positive_embedding = self.encoder(x)
        anchor_embedding = self.tanh(self.layer_norm(anchor_embedding))
        positive_embedding = self.tanh(self.layer_norm(positive_embedding))

        sparse_labels = torch.arange(anchor_embedding.size(0), device=anchor_embedding.device)

        similarities = self.similarity(anchor_embedding, positive_embedding)

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

    def test_step(self, x, batch_idx):
        anchor_embedding, positive_embedding = self.encoder(x)
        anchor_embedding = self.tanh(self.layer_norm(anchor_embedding))
        positive_embedding = self.tanh(self.layer_norm(positive_embedding))

        sparse_labels = torch.arange(anchor_embedding.size(0), device=anchor_embedding.device)

        similarities = self.similarity(anchor_embedding, positive_embedding)

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)