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

        image = Image.open(f'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/images').convert('RGB')
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_image_tensor = self.to_tensor(cropped_image)

        bbox = [xmin, ymin, xmax, ymax]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return cropped_image_tensor, bbox_tensor, label

if __name__ == '__main__':
    annotations_file = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv'
    images_folder = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data'
    
    dataset = dataloader_normal(annotations_file, images_folder)
    example_item = dataset[0]
    logging.debug(example_item)

class dataloader_binairy_image:
    """ Simple dataloader which return binary images"""

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
