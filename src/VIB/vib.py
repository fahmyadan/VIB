import torch 
from VAEs.models import VAE
from torchvision.models import inception_v3


class VIB(VAE):

    def __init__(self, in_channels, latent_dim, vib_h=299, vib_w=299,out_dims=1000,  hidden_dims = None, **kwargs):
        super().__init__(in_channels, latent_dim, hidden_dims, **kwargs)

        self.feat_extractor = inception_v3(pretrained=True)
        print('check')