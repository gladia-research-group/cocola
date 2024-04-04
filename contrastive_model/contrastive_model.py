import lightning as L
import torch
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F


class EfficientNetEncoder(torch.nn.Module):
    def __init__(self, drop_connect_rate: float):
        super(EfficientNetEncoder, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.efficientnet(x)

        y = x.squeeze(3).squeeze(2)

        return y


class CoCola(L.LightningModule):
    def __init__(self,
                 dropout_p: float = 0.1,
                 learning_rate: float = 0.001,
                 embedding_dim: int = 512):
        super().__init__()
        self.save_hyperparameters()

        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        self.dropout = torch.nn.Dropout(p=self.dropout_p)

        self.encoder = EfficientNetEncoder(drop_connect_rate=dropout_p)

        self.projection = torch.nn.Linear(1280, self.embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.linear = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def forward(self, x):
        anchor, positive = x["anchor"], x["positive"]

        anchor = self.dropout(self.encoder(anchor))
        anchor = self.dropout(self.projection(anchor))
        anchor = self.dropout(torch.tanh(self.layer_norm(anchor)))

        positive = self.dropout(self.encoder(positive))
        positive = self.dropout(self.projection(positive))
        positive = self.dropout(torch.tanh(self.layer_norm(positive)))

        anchor = self.linear(anchor)

        return anchor, positive

    def training_step(self, x, batch_idx):
        anchor_embedding, positive_embedding = self(x)

        sparse_labels = torch.arange(anchor_embedding.size(0), device=anchor_embedding.device)

        similarities = torch.mm(anchor_embedding, positive_embedding.t())

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, x, batch_idx):
        anchor_embedding, positive_embedding = self(x)

        sparse_labels = torch.arange(anchor_embedding.size(0), device=anchor_embedding.device)

        similarities = torch.mm(anchor_embedding, positive_embedding.t())

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

    def test_step(self, x, batch_idx):
        anchor_embedding, positive_embedding = self(x)

        sparse_labels = torch.arange(anchor_embedding.size(0), device=anchor_embedding.device)

        similarities = torch.mm(anchor_embedding, positive_embedding.t())

        loss = F.cross_entropy(similarities, sparse_labels)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == sparse_labels).double().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)