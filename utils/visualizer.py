import os
from scipy import stats
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .vis_utils import read_loss_from_file, sort_list_by_other, add_labels
import seaborn as sns
import matplotlib.pyplot as plt

TRAIN_FILE = "train_losses.log"

class Visualizer():
    def __init__(self, model, model_dir, max_traversal=0.475, upsample_factor=1):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.
        Parameters
        ----------
        model: disvae.vae.VAE
        model_dir: str
            The directory that the model is saved to and where the images will
            be stored.
        max_traversal: float, optional
            A percentage of the distribution (quantile), the maximum displacement 
            induced by a latent traversal. Symmetrical traversals are assumed.
            E.g. for the prior the distribution is a standard normal so `m=0.45` c
            orresponds to an absolute value of `1.645` because `2m=90%%` of a
            standard normal is between `-1.645` and `1.645`. Note in the case
            of the posterior, the distribution is not standard normal anymore.
        upsample_factor: floar, optional
            Scale factor to upsample the size of the tensor
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_dim = self.model.latent_dim
        self.max_traversal = max_traversal
        self.model_dir = model_dir
        self.upsample_factor = upsample_factor
        self.losses = read_loss_from_file(
            os.path.join(self.model_dir, TRAIN_FILE), "kl_loss_"
        )

    def reconstruct(self, data, name="", save=False):
        """Generate reconstructions of data through the model.
        Parameters
        ----------
        data: torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)
        name: str
            prefix of output name
        save: bool, optional
            Whether saving the image.
        """
        n_samples = len(data)
        with torch.no_grad():
            originals = data.to(self.device)
            recs, _, _ = self.model(originals)

        originals = originals.cpu()
        recs = recs.view(-1, *self.model.img_size).cpu()

        to_plot = torch.cat([originals, recs])
        filename = name + "_recons" if name else "_recons"
        return self._save_or_return(
            to_plot, (2, n_samples), filename=filename, save=save
        )

    def traversals(
            self, data=None, name="", n_per_latent=10, 
            n_latents=None, reorder_latents=False, save=False
        ):
        """Plot traverse through all latent dimensions (prior or posterior) one
        by one and plots a grid of images where each row corresponds to a latent
        traversal of one latent dimension.
        Parameters
        ----------
        data: bool, optional
            Data to use for computing the latent posterior. If `None` traverses
            the prior.
        name: str
            prefix of output name
        n_per_latent: int, optional
            The number of points to include in the traversal of a latent dimension.
        n_latents: int, optional
            The number of latent dimensions to display. 
            If `None` uses all latents.
        reorder_latents: bool, optional
            If the latent dimensions should be reordered or not
        save: bool, optional
            Whether saving the image.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim
        latent_samples = [
            self._traverse_one_latent(dim, n_per_latent, data=data) for dim in range(self.model.latent_dim)
        ]
        decoded_traversal = self._decode_latents(torch.cat(latent_samples, dim=0))

        if reorder_latents:
            n_images, *other_shape = decoded_traversal.size()
            n_rows = n_images // n_per_latent
            decoded_traversal = decoded_traversal.reshape(n_rows, n_per_latent, *other_shape)
            decoded_traversal = sort_list_by_other(decoded_traversal, self.losses)
            decoded_traversal = torch.stack(decoded_traversal, dim=0)
            decoded_traversal = decoded_traversal.reshape(n_images, *other_shape)

        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents), ...]

        size = (n_latents, n_per_latent)
        sampling_type = "{}_prior".format(name) if data is None else "{}_posterior".format(name)
        filename = "{}_{}".format(sampling_type, "traversals")

        return self._save_or_return(
            decoded_traversal, size, filename=filename, save=save
        )
    
    def traversals_original(
            self, dataset, percentile=1, n_per_latent=5, n_latents=None, name="", save=False
        ):
        """Plot traverse through all latent dimensions one by one and plots a grid of 
        original images based on corresponding value (high, low) of one latent dimension.
        Parameters
        ----------
        data: torch Dataset
        percentile: int
            Top and bottom percentile of data to be selected to plot
        n_per_latent: int, optional
            The number of points to include in the traversal of a latent dimension.
        n_latents: int, optional
            The number of latent dimensions to display. 
            If `None` uses all latents.
        name: str
            prefix of output name
        save: bool, optional
            Whether saving the image.
        """
        latent_path = os.path.join(self.model_dir, "data_latent.csv")
        latent = pd.read_csv(latent_path, header=None).values
        n_latents = n_latents if n_latents is not None else latent.shape[1]
        
        index = [self._traverse_one_latent_original(
            idx=dim, latent=latent, n_samples=n_per_latent, percentile=percentile
        ) for dim in range(n_latents)]
        
        to_plot = torch.stack([dataset[idx][0] for feat in index for idx in feat])
        filename = "{}_{}".format(name, "original_traversals")
        self._save_or_return(
            to_plot, size=(latent.shape[1], n_per_latent * 2), save=save, filename=filename
        )
    
    def latent_diagnosis(self):
        """
        Diagnosis of learned latent representation. Plot 
        1) The mean and log variance of all latent features.
        2) Correlation heatmap of latent features.
        """
        latent_path = os.path.join(self.model_dir, "data_latent.csv")
        latent = pd.read_csv(latent_path, header=None).values
        # plot mean, variance
        m, logv = np.mean(latent, axis=0), np.log(np.var(latent, axis=0))
        m_logv = pd.DataFrame(dict(feat=list(range(0, len(m))), mean=m, logv=logv))
        m_logv = m_logv.melt(id_vars=["feat"], var_name="latent")
        # correlation
        corr = np.round(np.corrcoef(latent.T), 3)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # plot
        f, ax = plt.subplots(1, 2, figsize=(15, 6))
        sns.barplot(x="feat", y="value", hue="latent", data=m_logv, edgecolor="k", ax=ax[0])
        sns.heatmap(corr, annot=True, mask=mask, cmap="vlag", ax=ax[1])
        fig_path = os.path.join(self.model_dir, "latent_diagnosis.png")
        f.savefig(fig_path)


    def _get_traversal_range(self, mean=0, std=1):
        """Return the corresponding traversal range in absolute terms."""
        max_traversal = self.max_traversal
        assert 0 < max_traversal < 0.5, "max_traversal should be within (0, 0.5)"
        max_traversal = (1 - 2 * max_traversal) / 2  # get percentiel left bound
        max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)  # get value left bound
        return (max_traversal, -1 * max_traversal) # symmetrical traversals
    
    def _traverse_one_latent(self, idx, n_samples, data=None):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx.
        Parameters
        ----------
        idx : int
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others are fixed.
        n_samples : int
            Number of samples to generate.
        data: torch.Tensor or None, optional
            Data to use for computing the posterior. Shape (N, C, H, W). If
            `None` then use the mean of the prior (all zeros) for all other dimensions.
        """
        if data is None:
            # mean of prior for other dimensions
            samples = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(), steps=n_samples)
        else:
            if data.size(0) > 1:
                raise ValueError("Only sample from the posterior of one images, but {} given.".format(data.size(0)))
            with torch.no_grad():
                post_mean, post_logvar = self.model.encoder(data.to(self.device))
                samples = self.model.reparameterize(post_mean, post_logvar)
                samples = samples.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.cpu()[0, idx]
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]
            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(
                *self._get_traversal_range(mean=post_mean_idx, std=post_std_idx), steps=n_samples
                # *self._get_traversal_range(mean=0, std=post_std_idx), steps=n_samples
            )
        for i in range(n_samples):
            samples[i, idx] = traversals[i]

        return samples
    
    def _traverse_one_latent_original(self, idx, n_samples, latent, percentile=1):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx.
        Parameters
        ----------
        idx: int
            index of latent features to visualize
        n_samples: int
            number of data samples to visualize
        latent: tensor
            all data latent matrix
        percentile: int
            Top and bottom percentile of data to be selected to plot
        """
        latent_v = latent[:,idx]
        # sample low value in latent space
        low = np.percentile(latent_v, percentile)
        low_index = np.where(latent_v < low)[0]
        low_index = self._sample_one_latent(latent, low_index, idx, n_samples)
        # sample high value in latent space
        high = np.percentile(latent_v, 100-percentile)
        high_index = np.where(latent_v > high)[0]
        high_index = self._sample_one_latent(latent, high_index, idx, n_samples)
        return np.concatenate((low_index, high_index))
    
    def _sample_one_latent(self, latent, index_to_slice, idx, n_samples, top=60):
        """
        Sample data whose selected latent features are in the top or bottom of all data distribution, 
        while the other latent features are close.
        Parameters
        ----------
        latent: tensor
            all data latent matrix
        index_to_slice:
            index of samples (rows) that are selected whose latent feature 
            are in the top or bottom of all data distribution
        idx: int
            index of latent feature to visualize
        n_samples: int
            number of data samples to visualize
        top: int
            top data samples to keep with rest of latent features being close
        """
        latent_other = np.delete(latent[index_to_slice,:], idx, axis=1)
        latent_other = np.abs(latent_other - np.mean(latent_other, axis=0))
        similar_sort = np.argsort(np.sum(latent_other, axis=1))
        return np.random.choice(index_to_slice[similar_sort][:top], size=n_samples, replace=False)
    
    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.
        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples).cpu()
    
    def _save_or_return(self, data, size, background=1, save=False, filename=None):
        """Create plot and save or return it."""
        to_plot = F.interpolate(data, scale_factor=self.upsample_factor)
        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError("Wrong size {} for datashape {}".format(size, to_plot.shape))
        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1], pad_value=background) # 0 for black, 1 for white
        if save and filename is not None:
            filepath = os.path.join(self.model_dir, filename + ".png")
            save_image(to_plot, filepath, **kwargs)
        else:
            return make_grid_img(to_plot, **kwargs)