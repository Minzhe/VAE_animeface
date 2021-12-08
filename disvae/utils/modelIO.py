import json
import os
import re
import numpy as np
import torch
from disvae import init_vae_model

MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"

def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.
    directory : str
        Path to the directory where to save the data.
    metadata : dict
        Metadata to save.
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        # save the minimum required for loading
        metadata = dict(
            img_size=model.img_size, 
            latent_dim=model.latent_dim,
            model_class=model.model_class,
            model_struc=model.model_struc
        )
    save_metadata(metadata, directory)

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)
    model.to(device)  # restore device

def load_model(directory, is_gpu=False, filename=MODEL_FILENAME):
    """Load a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.
    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")
    path_to_model = os.path.join(directory, MODEL_FILENAME)

    metadata = load_metadata(directory)
    img_size = metadata["img_size"]
    latent_dim = metadata["latent_dim"]
    model_type = metadata["model_type"]

    path_to_model = os.path.join(directory, filename)
    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
    return model

def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """Save the metadata to a training directory.

    Parameters
    ----------
    metadata:
        Object to save
    directory: string
        Path to folder where to save model. For example './experiments/mnist'.
    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)
    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)

def load_metadata(directory, filename=META_FILENAME):
    """Load the metadata of a training directory.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.
    """
    path_to_metadata = os.path.join(directory, filename)
    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)
    return metadata

def _get_model(model_type, img_size, latent_dim, device, path_to_model):
    """ Load a single model.

    Parameters
    ----------
    model_type : str
        The name of the model to load. For example Burgess.
    img_size : tuple
        Tuple of the number of pixels in the image width and height.
        For example (64, 64).
    latent_dim : int
        The number of latent dimensions in the bottleneck.
    device : str
        Either 'cuda' or 'cpu'
    path_to_device : str
        Full path to the saved model on the device.
    """
    model = init_vae_model(model_type, img_size, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model), strict=False)
    model.eval()
    return model