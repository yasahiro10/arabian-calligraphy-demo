import torch
import unittest
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.DEBUG)

class dataloader_normal(Dataset):
    def __init__(self, data_file, images_folder):
        self.data = self.load_data(data_file)
        self.images_folder = images_folder

    def __getitem__(self, index):
        image_filename = self.data['images'][index]
        bbox_list = self.data['bbox'][index]
        label = self.data['label'][index]

        image = self.load_image(image_filename)
        cropped_images = [self.crop_image(image, box) for box in bbox_list]
        bbox_tensors = [torch.tensor(box, dtype=torch.float32) for box in bbox_list]

        return {'images': cropped_images, 'bbox': bbox_tensors, 'label': label}

    def __len__(self):
        return len(self.data['images'])

    def load_data(self, data_file):
        data = {'images': [], 'bbox': [], 'label': []}
        with open(data_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(',')
                filename = parts[0]
                bbox = list(map(int, parts[4:8]))
                label = parts[3]
                data['images'].append(filename)
                data['bbox'].append(bbox)
                data['label'].append(label)
        return data

    def load_image(self, filename):
        image_path = f'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv'
        image = Image.open(image_path).convert('RGB')
        return transforms.ToTensor()(image)

    def crop_image(self, image, bbox):
        xmin, ymin, xmax, ymax = bbox
        cropped_image = image[:, ymin:ymax, xmin:xmax]
        return cropped_image
class dataloader_binairy_image:
    """ Simple dataloader which return binary images"""

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
