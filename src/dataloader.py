import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from src.utils.setup_logger import logger
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class dataloader_normal():
    """ Simple dataloader which return original images"""

    def __init__(self):
        image_tensor = []
        xy = pd.read_csv('C:/Users/msi/PycharmProjects/arabian-calligraphy-demo/data/train/annotations.csv',
                         delimiter=",")
        L = xy.iloc[:, 0].tolist()  # Assuming the first column is for x data
        for i in L:
            image = Image.open('C:/Users/msi/PycharmProjects/arabian-calligraphy-demo/data/train/{}'.format(i))
            to_tensor = transforms.ToTensor()
            image_tensor.append(to_tensor(image))
        self.x = image_tensor
        self.y = xy.iloc[:, 3]
        self.z = xy.iloc[:, 4:].values  # Convert DataFrame to NumPy array because the other columns
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        x_item = self.x[index]
        y_item = self.y[index]
        z_item = self.z[index]
        logger.debug(x_item)
        logger.debug(y_item)
        return x_item, y_item, z_item

    def __len__(self):
        return self.n_samples


class dataloader_binairy():
    """ Simple dataloader which return binary images"""
    def __init__(self):
        image_tensor = []
        xy = pd.read_csv('C:/Users/msi/PycharmProjects/arabian-calligraphy-demo/data/train/annotations.csv',
                         delimiter=",")
        L = xy.iloc[:, 0].tolist()  # Assuming the first column is for x data
        for i in L:
            image_tn = Image.open('C:/Users/msi/PycharmProjects/arabian-calligraphy-demo/data/train/{}'.format(i))
            image_np = np.array(image_tn)
            image_gray=cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
            # Convert the numpy array to a format compatible with OpenCV
            image = np.squeeze(image_gray)  # Remove any single-dimensional dimensions
            image = (image * 255).astype(np.uint8)
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            to_tensor = transforms.ToTensor()
            image_tensor.append(to_tensor(binary_image))
        self.x = image_tensor
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        x_item = self.x[index]
        return x_item

    def __len__(self):
        return self.n_samples
