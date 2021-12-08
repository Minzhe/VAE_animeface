import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from utils.datasets import img_dataset
import torch
from torch.utils.data import DataLoader
from disvae import init_vae_model, VAEGANTrainer, Evaluator
from disvae.models.losses import get_loss_f
from disvae.utils.modelIO import save_model, load_model
from utils.visualizer import Visualizer
from utils.vis_utils import get_samples
torch.autograd.set_detect_anomaly(True)

img_dir = "/home/mzzhang/data/animeface"
save_dir = "/home/mzzhang/project/VAE_animeface/results/vaegan_mae_betaH_loss"
formatter = logging.Formatter("%(asctime)s %(levelname)s - %(funcName)s: %(message)s", "%H:%M:%S")
model_struc = "Burgess"
img_size = (3,64,64)
latent_dim = 10
lr = 0.005
loss_f = dict(
    loss_name="VAEGAN", rec_dist="laplace", betaH_B=4, betaH_D=25, 
    kl_steps_anneal=50, bce_steps_anneal=50
)

# set logger
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
stream = logging.StreamHandler()
stream.setLevel("INFO")
stream.setFormatter(formatter)
logger.addHandler(stream)

# load data
anime_data = img_dataset(root=img_dir, size=img_size[1:])
data_loader = DataLoader(anime_data, batch_size=256, shuffle=True)

# initialize model
model = init_vae_model("vaegan", model_struc, img_size, latent_dim)
loss_f = get_loss_f(**loss_f)
# train model
trainer = VAEGANTrainer(
    model, optimizer="Adam", lr=lr, loss_f=loss_f,
    logger=logger, save_dir=save_dir, is_progress_bar=False
)
trainer.train(data_loader, epochs=100)
save_model(model, save_dir, filename="model.pt")
# # load model
# model = load_model(save_dir, filename="model.pt")
# # evaluate model
# evaluator = Evaluator(
#     model, loss_f=loss_f, logger=logger, save_dir=save_dir, is_progress_bar=True
# )
# # evaluator.compute_latent(data_loader)

# # visulization
# viz = Visualizer(model=model, model_dir=save_dir)
# # diagnosis
# viz.latent_diagnosis()
# # reconstruct
# data_sample = get_samples(data_loader, 8)
# viz.reconstruct(data_sample, name="test", save=True)
# # latent traversal
# viz.traversals(data=None, n_per_latent=10, n_latents=10, reorder_latents=True, name="test", save=True)
# # latent traversal original
# viz.traversals_original(anime_data, name="test", save=True)
