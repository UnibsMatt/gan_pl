import pytorch_lightning as pl
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class MnistDm(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.training_dataset = None
        self.test_dataset = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def prepare_data(self) -> None:
        self.training_dataset = MNIST("../", train=True, download=True, transform=self.transform)
        self.test_dataset = MNIST("../", train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=64, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=8)
