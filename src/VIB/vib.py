import torch 
from VAEs.models import VAE
from pytorch_lightning import LightningModule
from torchvision.models import inception_v3
import torch.nn.functional as F
from torch import nn


class VAE1(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, hidden_dims=None, out_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder (classifier head)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, out_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar


class VIB(LightningModule):
    def __init__(self, model_params, opt_params, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.vae = VAE(**model_params)
        self.opt_params = opt_params

    def kl_divergence(self, mu, logvar):
        # KL between N(mu, sigma) and N(0, 1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, mu, logvar = self.vae(x)
        ce_loss = F.cross_entropy(logits, y)
        kl = self.kl_divergence(mu, logvar)
        loss = ce_loss + self.opt_params.beta * kl

        self.log("train/loss", loss)
        self.log("train/ce_loss", ce_loss)
        self.log("train/kl", kl)

        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        logits, mu, logvar = self.vae(x)
        ce_loss = F.cross_entropy(logits, y)
        kl = self.kl_divergence(mu, logvar)
        loss = ce_loss + self.opt_params.beta * kl
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val/loss", loss)
        self.log("val/acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opt_params.LR, weight_decay=self.opt_params.weight_decay)

