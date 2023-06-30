import os

from lightning import pytorch as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import OneShotSimilarityPredictor
from utils import ContrastiveDataset

DATA_ROOT = os.environ.get("DATA_ROOT", "data")


def get_dataloader(data_root_dir: str, training: bool, batch_size: int) -> DataLoader:
    dataset = ContrastiveDataset(
        datasets.Omniglot,
        data_root_dir,
        background=training,
        transform=transforms.ToTensor(),
        download=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    model = OneShotSimilarityPredictor()
    train_dataloader = get_dataloader(DATA_ROOT, training=True, batch_size=128)
    trainer = pl.Trainer(accelerator="cpu")
    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()
