import unittest
from matplotlib import pyplot as plt
import pandas as pd
from src.dataloader import dataloader_normal,dataloader_binairy,dataloader_augmented
from src.utils.setup_logger import logger


class TestDataLoader(unittest.TestCase):

    def test_length_annotation(self):
        annotation = pd.read_csv("data/train/annotations.csv")
        dataset = dataloader_normal()
        logger.debug(dataset[0])

        plt.imshow(dataset.data["cropped_bbox"][0])
        plt.show()
        # logger.debug(dataset["label"])
        self.assertEqual(annotation.shape[0], len(dataset.data["cropped_bbox"]))
        self.assertEqual(annotation.shape[0], len(dataset.data["bbox"]))
        self.assertEqual(annotation.shape[0], len(dataset.data["label"]))
    def test_binary(self):
        annotation = pd.read_csv("data/train/annotations.csv")
        dataset = dataloader_binairy()
        # logger.debug(dataset)
        # self.assertEqual(annotation.shape[0], len(dataset["cropped_bbox"]))
        # self.assertEqual(annotation.shape[0], len(dataset["bbox"]))
        # self.assertEqual(annotation.shape[0], len(dataset["label"]))
        plt.imshow(dataset.data["cropped_bbox"][0],cmap='gray')
        plt.show()

    def test_dataloader_length(self):

        dataset = dataloader_normal()
        first_data = dataset[0]
        logger.debug(first_data)
        logger.debug(first_data[0])
        plt.imshow(first_data[0])
        plt.show()

    def test_dataloader_augmented(self):
        dataset = dataloader_augmented(["Gaussian blur","ColorJitter","GaussianNoise"],0,23,(106.67, 106.67),(2,2),4,1,0.1)
        logger.debug(len(dataset.data["cropped_bbox"]))
        plt.imshow(dataset.data["cropped_bbox"][0])
        plt.show()




        #plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
        # #Subplot 1: Rotated Image
        #plt.subplot(2, 2, 1)
        #plt.imshow(dataset.data["cropped_bbox"][0])
        #plt.title("Rotated Image")
        #plt.axis('off')  # Turn off axis labels

        # #Subplot 2: Blurred Image
        #plt.subplot(2, 2, 2)
        #plt.imshow(dataset.data["cropped_bbox"][1])
        #plt.title("Blurred Image")
        #plt.axis('off')  # Turn off axis labels

        # #Subplot 3: Brightness and Contrast Adjusted Image
        #plt.subplot(2, 2, 3)
        #plt.imshow(dataset.data["cropped_bbox"][2])
        #plt.title("Brightness and Contrast Adjusted Image")
        #plt.axis('off')  # Turn off axis labels

        ## Subplot 4: Image with Added Noise
        #plt.subplot(2, 2, 4)
        #plt.imshow(dataset.data["cropped_bbox"][3])
        #plt.title("Image with Added Noise")
        #plt.axis('off')  # Turn off axis labels

        ## Adjust layout for spacing between subplots
        #plt.tight_layout()

        # Show the figure with all images
        plt.show()

