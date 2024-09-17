import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision.io import read_image 
from PIL import Image
import torchvision.transforms as transforms
import scipy.io
from pathlib import Path
import glob
import random


class LandscapeDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, datatype='train'):
        self.transform = transforms_
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, datatype+'A') + '/*'))
        self.files_B = sorted(glob.glob(os.path.join(root, datatype+'B') + '/*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))