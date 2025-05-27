import torch 
from VAEs.models import VAE
from pytorch_lightning import LightningModule
from torchvision.models import inception_v3
import torch.nn.functional as F


class VIB(LightningModule):
    def __init__(self, model_params, opt_params, **kwargs):
        super().__init__()
        self.vae = VAE(**model_params)
        self.feat_extractor = inception_v3(pretrained=True)
        self.opt_params = opt_params

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, idx):

        x, y = batch

        pass 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt_params.LR, weight_decay=self.opt_params.weight_decay)
        return optimizer

# class VIB(VAE):

#     def __init__(self, in_channels, latent_dim, vib_h=299, vib_w=299,out_dims=1000,  hidden_dims = None, **kwargs):
#         super().__init__(in_channels, latent_dim, hidden_dims, **kwargs)

#         self.feat_extractor = inception_v3(pretrained=True)
        
