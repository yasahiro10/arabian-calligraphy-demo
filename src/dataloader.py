import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

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
            # TODO: apply transopose before putting image into list DONE
            image_tensor.append(to_tensor(cropped_image).numpy().transpose(1, 2, 0))

        self.data = {
            "cropped_bbox": image_tensor,
            "bbox": calligraphy_data.iloc[:,4:].values,
            "label": calligraphy_data.iloc[:,3]
        }

    def __repr__(self):
        return repr(self.data)
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data["label"])

class dataloader_binairy():
    """ Simple dataloader which return binary images"""
    def __init__(self):
        image_tensor = []
        to_tensor = transforms.ToTensor()
        calligraphy_data= pd.read_csv('data/train/annotations.csv',
                         delimiter=",")
        L = calligraphy_data.iloc[:, 0].tolist()  # Assuming the first column is for x data
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

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data["label"])
