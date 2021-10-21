import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from disvae.models.losses import get_loss_f
from utils.vis_utils import read_loss_from_file

TRAIN_FILE = "train_losses.log"

class Evaluator():
    def __init__(
        self, model, loss_f,
        device=torch.device("cpu"), logger=logging.getLogger(__name__),
        save_dir="results", is_progress_bar=True
    ):
        """
        Class to handle training of model.
        Parameters
        ----------
        model: disvae.vae.VAE
        loss_f: disvae.models.BaseLoss
            Loss function.
        device: torch.device, optional
            Device on which to run the code.
        logger: logging.Logger, optional
            Logger.
        save_dir : str, optional
            Directory for saving logs.
        is_progress_bar: bool, optional
            Whether to use a progress bar for training.
        """
        self.device = device
        self.loss_f = loss_f
        self.model = model.to(self.device)
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info("Testing Device: {}".format(self.device))
        self.losses = read_loss_from_file(
            os.path.join(self.save_dir, TRAIN_FILE), "kl_loss_"
        )

    def __call__(self, data_loader):
        """Compute all test losses.
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        losses = None
        self.logger.info("Computing losses...")
        losses = self.compute_losses(data_loader)
        self.logger.info("Losses: {}".format(losses))
        save_metadata(losses, self.save_dir, filename=TEST_LOSSES_FILE)

        if is_still_training:
            self.model.train()

        self.logger.info("Finished evaluating after {:.1f} min.".format((default_timer() - start) / 60))

        return losses
    
    def compute_losses(self, dataloader):
        """Compute all test losses.
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        storer = defaultdict(list)
        for data, index in tqdm(dataloader, leave=False, disable=not self.is_progress_bar):
            data = data.to(self.device)
            recon_batch, latent_dist, latent_sample = self.model(data)
            _ = self.loss_f(
                data, recon_batch, latent_dist, self.model.training, storer
            )
            losses = {k: sum(v) / len(dataloader) for k, v in storer.items()}
            return losses
    
    def compute_latent(self, dataloader):
        """Compute all test latent space.
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        latent_path = os.path.join(self.save_dir, "data_latent.csv")
        # compute latent
        index, latent = [], []
        for data, idx in tqdm(dataloader, leave=False, disable=not self.is_progress_bar):
            data = data.to(self.device)
            recon_batch, latent_dist, latent_sample = self.model(data)
            index.append(idx)
            latent.append(latent_sample)
        index = torch.cat(index).detach().numpy()
        latent = torch.cat(latent).detach().numpy()
        # reorder latent array
        latent = latent[np.argsort(index),:]
        latent = latent[:,np.argsort(self.losses)[::-1]]
        latent = round(pd.DataFrame(latent), 4)
        latent.to_csv(latent_path, header=False, index=False)