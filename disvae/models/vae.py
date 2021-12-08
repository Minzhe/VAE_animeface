#########################################################################
## vae.py
## Module containing the main VAE class. 
#########################################################################
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from .encoders import get_encoder
from .decoders import get_decoder
from .discriminator import get_discriminator

MODELS = ["Burgess"]
VAES = ["VAE", "VAEGAN"]

def init_vae_model(model_class, model_struc, img_size, latent_dim):
    """
    Return an instance of a VAE with encoder and decoder from `model_type`.

    Parameters
    -----------
    model_class: str
        class of model, "vae" or "vae-gan"
    model_strc: str
        model strcture, options in MODELS
    img_size: tuple of ints
        Size of images. E.g. (3, 64, 64).
    latent_dim: int
        latent dimension
    """
    model_struc = model_struc.lower().capitalize()
    model_class = model_class.upper()
    assert model_struc in MODELS, f"Unkown model_struc={model_struc}. Possible values: {MODELS}"
    assert model_class in VAES, f"Unknown model_class={model_class}. Possible values: {VAES}"
    # model structure
    encoder = get_encoder(model_struc)
    decoder = get_decoder(model_struc)
    # model class
    if model_class == "VAE":
        model = VAE(img_size, encoder, decoder, latent_dim)
    if model_class == "VAEGAN":
        discriminator = get_discriminator(model_struc)
        model = VAEGAN(
            discriminator, img_size=img_size, encoder=encoder, decoder=decoder, latent_dim=latent_dim
        )
    model.model_struc = model_struc  # store to help reloading
    model.model_class = model_class
    return model

class VAE(nn.Module):
    """
    Class of VAE model which defines model and forward pass.
    """
    def __init__(self, img_size, encoder, decoder, latent_dim):
        """
        Initialize VAE model.

        Parameters
        ----------
        img_size: tuple of ints
            Size of images. E.g. (3, 64, 64).
        encoder: nn.Module
            encoder model
        decoder: nn.Module
            decoder model
        latent_dim: int
            latent dimension
        """
        super(VAE, self).__init__()
        if list(img_size[1:]) != [64, 64]:
            raise RuntimeError("{} sized images not supported. Only (None, 64, 64) supported.".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. 
            Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. 
            Shape (batch_size, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.
        
        Parameters
        ----------
        x: torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x: torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class VAEGAN(VAE):
    """
    Class of VAE-GAN model which defines model and forward pass.
    """
    def __init__(self, discriminator, **kwargs):
        """
        Initialize VAE model.

        Parameters
        ----------
        discriminator: nn.Module
            discriminator module
        """
        super(VAEGAN, self).__init__(**kwargs)
        self.discriminator = discriminator(self.img_size, self.latent_dim)
    
    def forward(self, x):
        """
        Forward pass of model.
        
        Parameters
        ----------
        x: torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        batch_n = x.size()[0]
        # ----- VAE model ----- #
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        # ----- GAN model ------ #
        disc_x = torch.cat((x.detach().requires_grad_(), reconstruct), 0)
        disc_y = self.discriminator(disc_x)
        # print(disc_y)
        # print(disc_y.size())

        return reconstruct, latent_dist, latent_sample, disc_y[:batch_n], disc_y[batch_n:]