import argparse
import os 
from pathlib import Path
import logging 

import torch 
import torchvision
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from src.VIB import VIB
from src.lightning import ImageNetDataModule


def get_args():
    parser = argparse.ArgumentParser(description="Train a model using VIB.")

    # General
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name or ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Device
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to train on")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")

    # Model / Dataset
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset directory")

    # Logging / Checkpointing
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--save-model", action="store_true", help="Whether to save the best model")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Model checkpoint directory")

    # Evaluation
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation without training")

    args = parser.parse_args()
    return args


@hydra.main(config_path="src/VAEs/configs", config_name="base_config", version_base="1.3")
def train(cfg: DictConfig):
    # Example usage
    device = torch.device('cuda'if torch.cuda.is_available() else "cpu")
    # TODO: Add actual training logic here...
    model = VIB(model_params=cfg.model.model_params, opt_params=cfg.model.exp_params)

    datamodule = ImageNetDataModule(debug=True, **cfg.model.data_params)
    trainer = pl.Trainer(**cfg.trainer_params)

    trainer.fit(model=model, datamodule=datamodule)

    



if __name__ == "__main__":
    train()