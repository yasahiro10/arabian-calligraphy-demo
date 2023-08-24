import unittest
from matplotlib import pyplot as plt
import pandas as pd
from src.dataloader import dataloader_normal,dataloader_binairy
from src.utils.setup_logger import logger


class TestDataLoader(unittest.TestCase):

    def test_length_annotation(self):
        # TODO: dataset should contains something like that: returned by dataset
        #       { "cropped_bbox" : [tensors...],
        #         "bbox":  [[x,y,w,h]...],
        #         "label":[label1,..]
        #       }
        annotation = pd.read_csv("data/train/annotations.csv")
        dataset = dataloader_normal()
        logger.debug(dataset["label"][0])

        plt.imshow(dataset["cropped_bbox"][0])
        plt.show()
        # logger.debug(dataset["label"])
        # self.assertEqual(annotation.shape[0], len(dataset["cropped_bbox"]))
        # self.assertEqual(annotation.shape[0], len(dataset["bbox"]))
        # self.assertEqual(annotation.shape[0], len(dataset["label"]))
    def test_binary(self):
        annotation = pd.read_csv("data/train/annotations.csv")
        dataset = dataloader_binairy()
        # logger.debug(dataset)
        # self.assertEqual(annotation.shape[0], len(dataset["cropped_bbox"]))
        # self.assertEqual(annotation.shape[0], len(dataset["bbox"]))
        # self.assertEqual(annotation.shape[0], len(dataset["label"]))
        plt.imshow(dataset["cropped_bbox"][0],cmap='gray')
        plt.show()

    def test_dataloader_length(self):

        dataset = dataloader_normal()
        first_data = dataset[0]
        logger.debug(first_data)
        logger.debug(first_data[0].numpy().shape)
        plt.imshow(first_data[0].numpy().transpose(1, 2, 0))
        plt.show()

