import os

from lightning import pytorch as pl
from torch.utils.data import DataLoader

from dataset import OmniglotForVerificationTask
from model import OneShotSimilarityPredictor

DATASET_ROOT = os.environ.get("DATASET_ROOT", "data")


def create_dataloader(root_path: str, training: bool, batch_size: int) -> DataLoader:
    dataset = OmniglotForVerificationTask(
        root_path,
        train=training,
        n_samples=30_000,
        augment=False,
        download=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)


def main():
    model = OneShotSimilarityPredictor()
    train_dataloader = create_dataloader(DATASET_ROOT, training=True, batch_size=128)
    val_dataloader = create_dataloader(DATASET_ROOT, training=False, batch_size=128)
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=20)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
