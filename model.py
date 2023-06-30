import torch
from lightning import pytorch as pl
from torch import nn, optim, Tensor
from torch.nn import functional as F


def init_weights(module: nn.Module) -> None:
    """Initializes the model parameters (learnable weights and biases) of the
    neural network.

    Probability distributions to initialize from:

    * 𝒩(𝜇=0.0, 𝜎=1e-2) for the scaling weights of convolutional layers
    * 𝒩(𝜇=0.0, 𝜎=2e-1) for the scaling weights of fully-connected layers
    * 𝒩(𝜇=0.5, 𝜎=1e-2) for the biases of convolutional and fully-connected layers.

    All other weights and biases are left unchanged.

    :param (nn.Module) module: The neural network whose weights and biases to initialize
    """
    if type(module) == nn.Conv2d:
        nn.init.normal_(module.weight, mean=0, std=1e-2)
        nn.init.normal_(module.bias, mean=0.5, std=1e-2)
    elif type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=2e-1)
        nn.init.normal_(module.bias, mean=0.5, std=1e-2)


# The inline comments beside forward-propagation calls show the shape of the output
# tensor assuming the main Siamese network's input tensors each has the same shape as in
# the paper: (N, 1, 105, 105) where N is the batch size.


class DefaultEncoder(nn.Module):
    """The default encoder backend (a.k.a. feature extractor) that takes an image and
    outputs a 4096-dimensional vector."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)
        self.fc = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.conv1(x))  # (N, 64, 96, 96)
        x = F.max_pool2d(x, kernel_size=2)  # (N, 64, 48, 48)

        x = torch.relu(self.conv2(x))  # (N, 128, 42, 42)
        x = F.max_pool2d(x, kernel_size=2)  # (N, 128, 21, 21)

        x = torch.relu(self.conv3(x))  # (N, 128, 18, 18)
        x = F.max_pool2d(x, kernel_size=2)  # (N, 128, 9, 9)

        x = torch.relu(self.conv4(x))  # (N, 256, 6, 6)

        x = torch.flatten(x, start_dim=1)  # (N, 256 * 6 * 6)
        x = torch.sigmoid(self.fc(x))  # (N, 4096)

        return x


class DefaultSimilarityPredictor(nn.Module):
    """The default final layer used in the paper"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=4096, out_features=1)
        self.apply(init_weights)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        l1_dists = torch.abs(x1 - x2)  # (N, 4096)
        similarities = self.fc(l1_dists)  # (N, 1)

        # Return the raw similarity scores (without sigmoid) and use `bce_with_logits`
        # for increased numerical stability.
        return similarities


class SiameseNetworkForSimilarity(nn.Module):
    """A Siamese network that takes two images and predicts their similarity."""

    def __init__(
        self,
        encoder: nn.Module = DefaultEncoder(),
        predictor: nn.Module = DefaultSimilarityPredictor(),
        final_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.final_sigmoid = final_sigmoid

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        features1 = self.encoder(x1)  # (N, 4096)
        features2 = self.encoder(x2)  # (N, 4096)
        predictions = self.predictor(features1, features2)  # (N, 1)
        predictions = predictions.squeeze(dim=1)  # (N)
        if self.final_sigmoid:
            predictions = torch.sigmoid(predictions)
        return predictions


class OneShotSimilarityPredictor(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.network = SiameseNetworkForSimilarity()
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> dict[str, any]:
        x1, x2, y = batch
        y_hat = self.network(x1, x2)
        loss = self.criterion(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self) -> optim.Optimizer:
        # starting simple with SGD
        return optim.SGD(self.parameters(), lr=1e-3, momentum=0.5, weight_decay=0.05)
