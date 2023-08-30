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

class dataloader_augmented():

    def  __int__(self, augmentation_type, *params):
        """
TODO:
        :param augmentation_type: list of string contains the possible augmentation, it can be:
            - Rotation (without a big angle)
            - Gaussian blur
            - Contrast and brightness
            - Adding noise done
        :param params: params of the augmentation
        :return:
        """
        self.augmentation_type = augmentation_type
        self.params = params
        Rotate_Transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=params[0])
        ])
        Gaussian_transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=(params[1],params[2]), sigma=(params[3],params[4]))])
        Color_Transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(params[5],params[6]),contrast=params[7])
        ])
        AddGaussianNoise_Transformation= transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            AddGaussianNoise(params[8], params[9]),
            transforms.ToPILImage()
        ])

        image_tensor = []
        to_tensor = transforms.ToTensor()
        calligraphy_data = pd.read_csv('data/train/annotations.csv',
                                       delimiter=",")
        for index, row in calligraphy_data.iterrows():
            image_path = row['filename']
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            image_jpeg = Image.open('data/train/{}'.format(image_path))
            cropped_image = image_jpeg.crop((xmin, ymin, xmax, ymax))

            # Rotated_Img = Rotate_Transformation(cropped_image)
            # Gaussian_image = Gaussian_transformation(cropped_image)
            # color_image= Color_Transformation(cropped_image)
            # noise_image= AddGaussianNoise_Transformation(cropped_image)
            for item in augmentation_type:
                if item =="Rotation":
                    Rotated_Img = Rotate_Transformation(cropped_image)
                    image_tensor.append(to_tensor(Rotated_Img).numpy().transpose(1, 2, 0))
                if item == "Gaussian blur":
                    Gaussian_image = Gaussian_transformation(cropped_image)
                    image_tensor.append(to_tensor(Gaussian_image).numpy().transpose(1, 2, 0))
                if item ==  "ColorJitter":
                    color_image = Color_Transformation(cropped_image)
                    image_tensor.append(to_tensor(color_image).numpy().transpose(1, 2, 0))
                if item == "GaussianNoise":
                    noise_image= AddGaussianNoise_Transformation(cropped_image)
                    image_tensor.append(to_tensor(noise_image).numpy().transpose(1, 2, 0))
        self.data = {
            "cropped_bbox": image_tensor,
            "bbox": calligraphy_data.iloc[:, 4:].values,
            "label": calligraphy_data.iloc[:, 3]
            }
    def __getitem__(self, index):
        return self.data["cropped_bbox"][index], self.data["bbox"][index], self.data["label"][index]
    def __len__(self):
        return len(self.data["label"])
