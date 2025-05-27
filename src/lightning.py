import os
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
from torchvision.datasets import MNIST
import torch


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def setup(self, stage=None):
        self.mnist_train = MNIST(root=".", train=True, download=True, transform=self.transform)
        self.mnist_val = MNIST(root=".", train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = None,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 224,
        debug = False, 
        patch_size: int = None, 
    ):
        super().__init__()
        self.data_dir = data_path
        self.batch_size_train = train_batch_size
        self.batch_size_val = val_batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.debug = debug

    def setup(self, stage: Optional[str] = None):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean/std
            std=[0.229, 0.224, 0.225],
        )

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        if not self.debug:
            self.train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=train_transform
            )
            self.val_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "val"), transform=val_transform
            )
        else:
            num_classes = 10
            num_train = int(1e4)
            num_val = 500
            train_imgs = torch.randn((num_train, 3, self.image_size, self.image_size))
            val_images = torch.randn(num_val, 3, self.image_size, self.image_size)
            train_labels = torch.randint(0, num_classes, (num_train,))
            val_labels = torch.randint(0, num_classes, (num_val,))

            self.train_dataset = TensorDataset(train_imgs, train_labels)
            self.val_dataset = TensorDataset(val_images, val_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    


