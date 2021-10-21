import os
import glob
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class img_dataset(Dataset):
    def __init__(self, root, size):
        """
        Custom dataset

        Parameters
        ----------
        root: str
            Path to the dataset root.
        size: tuple of int
            Size (width, height) to rescale the images. If `None` don't rescale.
        """
        self.root = root
        imgs = []
        for ext in [".png", ".jpg", ".jpeg"]:
            imgs += glob.glob(os.path.join(self.root, '*' + ext))
        self.imgs = imgs
        self.size = size
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        width, height = img.size
        if width != self.size[1] or height != self.size[0]:
            img = img.resize(self.size, Image.ANTIALIAS)
        return self.transform(img), idx
