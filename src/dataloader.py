import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from src.utils.setup_logger import logger


class dataloader_normal():
    """ Simple dataloader which return original images"""
    def __init__(self):
        xy = pd.read_csv('C:/Users/msi/Desktop/arabian calligraphy dataset/train/annotations.csv', delimiter=",")
        self.x = xy.iloc[:, 0].tolist()# Assuming the first column is for x data
        self.y = xy.iloc[:, 1:].values # Convert DataFrame to NumPy array because the other columns
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        x_item = self.x[index]
        y_item = self.y[index]
        logger.debug(x_item)
        logger.debug(y_item)
        return x_item, y_item

    def __len__(self):
        return self.n_samples



class dataloader_binairy_image:
    """ Simple dataloader which return binary images"""

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

