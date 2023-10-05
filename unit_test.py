import unittest
from matplotlib import pyplot as plt
import pandas as pd
from src.dataloader import dataloader_normal,dataloader_binairy,dataloader_augmented
from src.models import VGG16,VGg16
from src.utils.setup_logger import logger
from torchviz import  make_dot
import torch
from IPython.display import display


import os

from train import train

os.environ["Path"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

class TestDataLoader(unittest.TestCase):

    def test_length_annotation(self):
        annotation = pd.read_csv("data/train/annotations.csv")
        dataset = dataloader_normal()
        logger.debug(dataset.data["label"])
       #plt.imshow(dataset.data["cropped_bbox"][0])
        #plt.show()
        logger.debug(tuple(dataset["label"]))
        #self.assertEqual(annotation.shape[0], len(dataset.data["cropped_bbox"]))
        #self.assertEqual(annotation.shape[0], len(dataset.data["bbox"]))
        #self.assertEqual(annotation.shape[0], len(dataset.data["label"]))
    def test_binary(self):
        annotation = pd.read_csv("data/train/annotations.csv")
        dataset = dataloader_binairy()
        logger.debug(dataset.data)
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
        # plt.imshow(first_data[0])
        # plt.show()

    def test_dataloader_augmented(self):
        dataset = dataloader_augmented(["Rotation","Gaussian blur","ColorJitter","GaussianNoise"],0,23,(106.67, 106.67),(2,2),4,1,0.1)
        # logger.debug(len(dataset.data["cropped_bbox"]))
        # logger.debug(len(dataset.data))
        logger.debug([i for i in enumerate(dataset.data)])
        logger.debug([i for i in dataset.data])
        logger.debug(type(dataset.data))
        logger.debug(dataset.data["cropped_bbox"][0])

        # plt.imshow(dataset.data["cropped_bbox"][0])
        # plt.show()
        #
        # plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
        # # #Subplot 1: Rotated Image
        # plt.subplot(2, 2, 1)
        # plt.imshow(dataset.data["cropped_bbox"][0])
        # plt.title("Rotated Image")
        # plt.axis('off')  # Turn off axis labels
        #
        # # #Subplot 2: Blurred Image
        # plt.subplot(2, 2, 2)
        # plt.imshow(dataset.data["cropped_bbox"][1])
        # plt.title("Blurred Image")
        # plt.axis('off')  # Turn off axis labels
        #
        # # #Subplot 3: Brightness and Contrast Adjusted Image
        # plt.subplot(2, 2, 3)
        # plt.imshow(dataset.data["cropped_bbox"][2])
        # plt.title("Brightness and Contrast Adjusted Image")
        # plt.axis('off')  # Turn off axis labels
        #
        # ## Subplot 4: Image with Added Noise
        # plt.subplot(2, 2, 4)
        # plt.imshow(dataset.data["cropped_bbox"][3])
        # plt.title("Image with Added Noise")
        # plt.axis('off')  # Turn off axis labels
        #
        # ## Adjust layout for spacing between subplots
        # plt.tight_layout()
        #
        # # Show the figure with all images
        # plt.show()

class TestModel(unittest.TestCase):
    def test_architecture(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchviz")
        model= VGG16()
        #logger.debug(model.layer13(model.layer12(model.layer11(model.layer10(model.layer9(model.layer8(model.layer7(model.layer6(model.layer5(model.layer4(model.layer3(model.layer2(model.layer1(torch.randn(1, 3, 224, 224)))))))))))))).shape)
        dummy_input = torch.randn(1, 3, 512,512)
        #with torch.no_grad():
        #    output = model(dummy_input)
        # Visualize the computation graph
        dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
        dot.render("vgg16_graph")
        display(dot)
        print(model.summary())


    def test_train(self):
        model = VGg16(18)
        train(model, epochs=1)
