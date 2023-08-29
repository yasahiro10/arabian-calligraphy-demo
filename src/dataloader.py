import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

class dataloader_normal(Dataset):
    def __init__(self, annotations_file, images_folder):
        self.annotations = pd.read_csv(annotations_file, delimiter=",")
        self.images_folder = images_folder
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations.iloc[index]
        image_path = annotation['filename']
        xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
        label = annotation['class']

        image = Image.open(f'{self.images_folder}/{image_path}').convert('RGB')
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_image_tensor = self.to_tensor(cropped_image)

        bbox = [xmin, ymin, xmax, ymax]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
           "images": cropped_image_tensor, 
            "bbox":bbox_tensor, 
            "label":label}


annotations_file = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv'
images_folder = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/images'
    
dataset = dataloader_normal(annotations_file, images_folder)
example_item = dataset[1]
logging.debug(example_item)

'''class dataloader_binairy_image(Dataset):
    """Simple dataloader which returns binary images"""

    def __init__(self, annotations_file, images_folder):
        self.annotations = pd.read_csv(annotations_file, delimiter=",")
        self.images_folder = images_folder
        self.to_binary = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x)))
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations.iloc[index]
        image_path = annotation['filename']
        xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
        label = annotation['class']

        image = Image.open(f'{self.images_folder}/{image_path}').convert('L')  
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_binary_image = self.to_binary(cropped_image)

        bbox = [xmin, ymin, xmax, ymax]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
           "images": cropped_image_tensor, 
            "bbox":bbox_tensor, 
            "label":label}


annotations_file = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv'
images_folder = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/images'
    
dataset = dataloader_binairy_image(annotations_file, images_folder)
example_item = dataset[0]
logging.debug(example_item)'''