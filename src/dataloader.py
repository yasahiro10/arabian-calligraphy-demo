import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch

class dataloader_normal():
    """ Simple dataloader which return original images"""

    def __init__(self):
        image_tensor = []
        to_tensor = transforms.ToTensor()
        calligraphy_data= pd.read_csv('data/train/annotations.csv', delimiter=",")
        for index,row in calligraphy_data.iterrows():
            image_path=row['filename']
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            image = Image.open('data/train/{}'.format(image_path))
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            image_tensor.append(to_tensor(cropped_image).numpy().transpose(1, 2, 0))

        self.data = {
            "cropped_bbox": image_tensor,
            "bbox": calligraphy_data.iloc[:,4:].values,
            "label": calligraphy_data.iloc[:,3]
        }

    def __getitem__(self, index):
        return self.data["cropped_bbox"][index], self.data["bbox"][index], self.data["label"][index]

    def __len__(self):
        return len(self.data["label"])

class dataloader_binairy():
    """ Simple dataloader which return binary images"""
    def __init__(self):
        image_tensor = []
        to_tensor = transforms.ToTensor()
        calligraphy_data= pd.read_csv('data/train/annotations.csv',
                         delimiter=",")
        for index, row in calligraphy_data.iterrows():
            image_path = row['filename']
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            image_jpeg= Image.open('data/train/{}'.format(image_path))
            cropped_image = image_jpeg.crop((xmin, ymin, xmax, ymax))
            image_np = np.array(cropped_image)
            image_gray=cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
            # Convert the numpy array to a format compatible with OpenCV
            image = np.squeeze(image_gray)  # Remove any single-dimensional dimensions
            image = (image * 255).astype(np.uint8)
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_tensor.append(to_tensor(binary_image).numpy().transpose(1, 2, 0))
        self.data = {
            "cropped_bbox": image_tensor,
            "bbox": calligraphy_data.iloc[:, 4:].values,
            "label": calligraphy_data.iloc[:, 3]
        }

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data["label"])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class dataloader_augmented:
    def  __init__(self, augmentation_type=[],degrees=0,kernel_size=0,sigma=(),brightness=(),contrast=0,mean=0,std=1):
        self.augmentation_type = augmentation_type
        self.degrees = degrees
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.brightness = brightness
        self.contrast = contrast
        self.mean = mean
        self.std=std
        Rotate_Transformation =transforms.RandomRotation(degrees= degrees)
        Gaussian_transformation =transforms.GaussianBlur(kernel_size = kernel_size, sigma=sigma)
        Color_Transformation =transforms.ColorJitter(brightness=brightness,contrast=contrast)

        AddGaussianNoise_Transformation= transforms.Compose([
             transforms.ToTensor(),
             AddGaussianNoise(mean,std),
             transforms.ToPILImage()
         ])
        image_tensor = []
        bbox_list=[]
        label_list=[]
        to_tensor = transforms.ToTensor()
        calligraphy_data = pd.read_csv('data/train/annotations.csv', delimiter=",")
        for index, row in calligraphy_data.iterrows():
            image_path = row['filename']
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            image_jpeg = Image.open('data/train/{}'.format(image_path))
            cropped_image = image_jpeg.crop((xmin, ymin, xmax, ymax))
            for item in augmentation_type:
                if item =="Rotation":
                    Rotated_Img = Rotate_Transformation(cropped_image)
                    image_tensor.append(to_tensor(Rotated_Img).numpy().transpose(1, 2, 0))
                    bbox_list.append([xmin, ymin, xmax, ymax])
                    label_list.append(row['class'])
                if item == "Gaussian blur":
                    Gaussian_image = Gaussian_transformation(cropped_image)
                    image_tensor.append(to_tensor(Gaussian_image).numpy().transpose(1, 2, 0))
                    bbox_list.append([xmin, ymin, xmax, ymax])
                    label_list.append(row['class'])
                if item ==  "ColorJitter":
                    color_image = Color_Transformation(cropped_image)
                    image_tensor.append(to_tensor(color_image).numpy().transpose(1, 2, 0))
                    bbox_list.append([xmin, ymin, xmax, ymax])
                    label_list.append(row['class'])
                if item == "GaussianNoise":
                    noise_image= AddGaussianNoise_Transformation(cropped_image)
                    image_tensor.append(to_tensor(noise_image).float().numpy().transpose(1, 2, 0))
                    bbox_list.append([xmin, ymin, xmax, ymax])
                    label_list.append(row['class'])
        self.data = {
            "cropped_bbox": image_tensor,
            "bbox": bbox_list,
            "label": label_list
            }
    def __getitem__(self, index):
        return self.data["cropped_bbox"][index], self.data["bbox"][index], self.data["label"][index]
    def __len__(self):
        return len(self.data["label"])
