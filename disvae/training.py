import logging
import os
import numpy as np
from timeit import default_timer
from collections import defaultdict
from tqdm import trange
import torch
from torch import optim
from torch.nn import functional as F
from disvae.utils.modelIO import save_model


TRAIN_LOSSES_LOGFILE = "train_losses.log"
OPTIMIZER = ["Adam"]

class VAETrainer():
    def __init__(
        self, model, optimizer, lr, loss_f, device=torch.device("cpu"), 
        logger=logging.getLogger(__name__), save_dir="results", is_progress_bar=True
    ):
        """
        Class to handle training of model.
        
        Parameters
        ----------
        model: disvae.vae.VAE
        optimizer: str
            optimizer
        lr: float
            learning rate
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
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = get_optimizer(self.model.parameters(), optimizer=optimizer, lr=lr)
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.loss_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.logger.info("Training Device: {}".format(self.device))

    def train(self, data_loader, epochs=100, checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        epochs: int, optional
            Number of epochs to train the model for.
        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            # train one epoch
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.logger.info(
                "Epoch: {} Average loss per image: {:.2f}".format(epoch + 1, mean_epoch_loss)
            )
            self.loss_logger.log(epoch, storer)
            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir, filename="model-{}.pt".format(epoch))

        self.model.eval()
        delta_time = (default_timer() - start) / 60
        self.logger.info("Finished training after {:.1f} min.".format(delta_time))

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        storer: dict
            Dictionary in which to store important variables for vizualisation.
        epoch: int
            Epoch number
        
        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(
            desc="Epoch {}".format(epoch + 1), 
            leave=False,
            disable=not self.is_progress_bar
        )
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, idx) in enumerate(data_loader):
                iter_loss = self._train_iteration(data, storer)
                epoch_loss += iter_loss
                t.set_postfix(loss=iter_loss)
                t.update()
        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        recon_batch, latent_dist, latent_sample = self.model(data)
        loss = self.loss_f(
            data=data, recon_data=recon_batch, 
            latent_dist=latent_dist, 
            is_train=self.model.training, 
            storer=storer
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class VAEGANTrainer():
    def __init__(
        self, model, optimizer, lr, loss_f, device=torch.device("cpu"), 
        logger=logging.getLogger(__name__), save_dir="results", is_progress_bar=True
    ):
        """
        Class to handle training of VAEGAN model.
        
        Parameters
        ----------
        optimizer_disc: torch.optim.Optimizer
        """
        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer_vae = get_optimizer(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), 
            optimizer=optimizer, lr=lr
        )
        self.optimizer_gan = get_optimizer(
            self.model.discriminator.parameters(), optimizer=optimizer, lr=lr
        )
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.loss_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.logger.info("Training Device: {}".format(self.device))

    def train(self, data_loader, epochs=100, checkpoint_every=10):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        epochs: int, optional
            Number of epochs to train the model for.
        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            # train one epoch
            storer = defaultdict(list)
            loss, gen_loss, disc_loss = self._train_epoch(data_loader, storer, epoch)
            message = "Epoch: {} Average loss per image: {:.2f}; ".format(epoch + 1, loss)
            message += "reconstruct loss: {:.2f}; generator loss: {:.2f}; discriminator loss: {:.2f}".format(loss - gen_loss, gen_loss, disc_loss)
            self.logger.info(message)
            self.loss_logger.log(epoch, storer)
            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir, filename="model-{}.pt".format(epoch))

        self.model.eval()
        delta_time = (default_timer() - start) / 60
        self.logger.info("Finished training after {:.1f} min.".format(delta_time))
    
    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        storer: dict
            Dictionary in which to store important variables for vizualisation.
        epoch: int
            Epoch number
        
        Return
        ------
        loss: float
            Mean vae reconstruction loss plus generator loss
        gen_loss: float
            Mean generator loss per image
        disc_loss: float
            Mean discriminator loss per image
        """
        loss, gen_loss, disc_loss = 0., 0., 0.
        kwargs = dict(
            desc="Epoch {}".format(epoch + 1), 
            leave=False,
            disable=not self.is_progress_bar
        )
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, idx) in enumerate(data_loader):
                iter_loss, iter_gen_loss, iter_disc_loss = self._train_iteration(data, storer)
                loss += iter_loss
                gen_loss += iter_gen_loss
                disc_loss += iter_disc_loss
                t.set_postfix(loss=iter_loss, gen_loss=iter_gen_loss, disc_loss=iter_disc_loss)
                t.update()
        n = len(data_loader)
        return loss / n, gen_loss / n, disc_loss / n
    
    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        recon_batch, latent_dist, latent_sample, y, recon_y = self.model(data)
        loss, gen_loss, disc_loss = self.loss_f(
            data=data, recon_data=recon_batch, latent_dist=latent_dist, 
            y=y, recon_y=recon_y, is_train=self.model.training, storer=storer
        )
        # discriminator loss
        self.optimizer_gan.zero_grad()
        disc_loss.backward(retain_graph=True)
        self.optimizer_gan.step()
        # vae resconstruction and generator loss combined
        self.optimizer_vae.zero_grad()
        loss.backward()
        self.optimizer_vae.step()
        # generator loss only for visulization
        

        return loss.item(), gen_loss.item(), disc_loss.item()

def get_optimizer(params, optimizer, lr):
    """
    Get optimizer
    """
    optimizer = optimizer.lower().capitalize()
    assert optimizer in OPTIMIZER, f"optimizer {optimizer} not found in {OPTIMIZER}"
    if optimizer == "Adam":
        return optim.Adam(params, lr=lr)

class LossesLogger(object):
    """
    Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """
    def __init__(self, file_path):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path):
            os.remove(file_path)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, round(np.mean(v), 4)])
            self.logger.debug(log_string)