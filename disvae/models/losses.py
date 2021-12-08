#########################################################################
## losses.py
## Module containing all vae losses. 
#########################################################################
import abc
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

LOSSES = ["VAE", "betaH", "betaB", "VAEGAN"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]

def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"], kl_steps_anneal=kwargs_parse["kl_steps_anneal"])
    if loss_name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(
            C_init=kwargs_parse["betaB_initC"], C_fin=kwargs_parse["betaB_finC"],
            gamma=kwargs_parse["betaB_G"], **kwargs_all
        )
    elif loss_name == "VAEGAN":
        return VAEGANLoss(
            beta=kwargs_parse["betaH_B"], delta=kwargs_parse["betaH_D"], 
            bce_steps_anneal=kwargs_parse["bce_steps_anneal"], **kwargs_all
        )
    else:
        assert loss_name not in LOSSES
        raise ValueError("Uknown loss: {}".format(loss_name))

class BaseLoss(abc.ABC):
    def __init__(self, record_loss_every=50, rec_dist="bernoulli", kl_steps_anneal=0):
        """
        Base class for losses.
        Parameters
        ----------
        record_loss_every: int, optional
            Every how many steps to record the loss.
        rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
            Reconstruction distribution istribution of the likelihood on the each pixel.
            Implicitely defines the reconstruction loss. Bernoulli corresponds to a
            binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
            corresponds to L1.
        kl_steps_anneal: nool, optional
            Number of annealing steps where gradually adding the kl regularisation.
        """
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.kl_steps_anneal = kl_steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.
        Parameters
        ----------
        data: torch.Tensor
            Input data (e.g. batch of images). 
            Shape: (batch_size, n_chan, height, width).
        recon_data: torch.Tensor
            Reconstructed data. 
            Shape: (batch_size, n_chan, height, width).
        latent_dist: tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).
        is_train: bool
            Whether currently in train mode.
        storer: dict
            Dictionary in which to store important variables for vizualisation.
        kwargs:
            Loss specific arguments
        """

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1
        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None
        return storer

class BetaHLoss(BaseLoss):
    def __init__(self, beta=4, **kwargs):
        """
        Compute the Beta-VAE loss as in [1]

        Parameters
        ----------
        beta : float, optional
            Weight of the kl divergence.
        kwargs:
            Additional arguments for `BaseLoss`, e.g. rec_dist`.
        References
        ----------
            [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
            a constrained variational framework." (2016).
        """
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data, storer=storer, distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        anneal_reg = linear_annealing(0, 1, self.n_train_steps, self.kl_steps_anneal) if is_train else 1
        loss = rec_loss + anneal_reg * (self.beta * kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss

class BetaBLoss(BaseLoss):
    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        r"""
        Compute the Beta-VAE loss as in [1]
        
        Parameters
        ----------
        C_init : float, optional
            Starting annealed capacity C.
        C_fin : float, optional
            Final annealed capacity C.
        gamma : float, optional
            Weight of the KL divergence term.
        kwargs:
            Additional arguments for `BaseLoss`, e.g. rec_dist`.
        References
        ----------
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data, storer=storer, distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        C = linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.kl_steps_anneal) if is_train else self.C_fin
        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer["loss"].append(loss.item())

        return loss

class VAEGANLoss(BaseLoss):
    def __init__(self, beta=4, delta=4, bce_steps_anneal=0, **kwargs):
        """
        Compute the VAE-GAN loss

        Parameters
        ----------
        beta: float
            Weight of the kl divergence.
        delta: float
            Weight of the bce loss of reconstructed images
        bce_steps_anneal: nool, optional
            Number of annealing steps where gradually adding the bce loss of reconstructed images.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.delta = delta
        self.bce_steps_anneal = bce_steps_anneal
    
    def __call__(self, data, recon_data, latent_dist, y, recon_y, is_train, storer):
        storer = self._pre_call(is_train, storer)

        # reconstruction loss & prior
        rec_loss = _reconstruction_loss(data, recon_data, storer=storer, distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        # discriminator loss & generator (decoder) loss
        gen_loss, disc_loss = _discriminator_loss(y, recon_y, storer)
        # combine reconstruction loss and generator loss
        anneal_reg = linear_annealing(0, 1, self.n_train_steps, self.kl_steps_anneal) if is_train else 1
        anneal_bce = linear_annealing(0, 1, self.n_train_steps, self.bce_steps_anneal) if is_train else 1
        loss = rec_loss + anneal_reg * (self.beta * kl_loss) + anneal_bce * (self.delta * gen_loss)

        if storer is not None:
            storer["loss"].append(loss.item())

        return loss, gen_loss, disc_loss

def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.
    
    Parameters
    ----------
    data: torch.Tensor
        Input data (e.g. batch of images). 
        Shape: (batch_size, n_chan, height, width).
    recon_data: torch.Tensor
        Reconstructed data. 
        Shape: (batch_size, n_chan, height, width).
    distribution: {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.
    storer: dict
        Dictionary in which to store important variables for vizualisation.
    
    Returns
    -------
    loss: torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))
    loss = loss / batch_size

    if storer is not None:
        storer["recon_loss"].append(loss.item())

    return loss

def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.
    
    Parameters
    ----------
    mean: torch.Tensor
        Mean of the normal distribution. 
        Shape (batch_size, latent_dim) where
        D is dimension of distribution.
    logvar: torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)
    storer: dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer["kl_loss"].append(total_kl.item())
        for i in range(latent_dim):
            storer["kl_loss_" + str(i)].append(latent_kl[i].item())

    return total_kl

def _discriminator_loss(y, recon_y, storer):
    """
    Calculate the binary cross entropy loss for discriminator.
    It seeks to maximize the log probability of real images and 
    the log of the inverse probability for fake images.

    Parameters
    ----------
    p_real: torch.Tensor
        probability to be real of original images.
    p_fake: torch.Tensor
        probability to be real of reconstructed images.
    """
    bce_original = -torch.log(y + 1e-3)
    bce_recon = -torch.log(1 - recon_y + 1e-3)
    gen_loss = torch.mean(bce_original)
    disc_loss = (gen_loss + torch.mean(bce_recon)) / 2

    if storer is not None:
        storer["gen_loss"].append(gen_loss.item())
        storer["disc_loss"].append(disc_loss.item())
        
    return gen_loss, disc_loss

def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed