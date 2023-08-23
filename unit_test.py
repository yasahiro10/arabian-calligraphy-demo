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
        #logger.debug(annotation["class"])
        dataset = dataloader_normal()
        #logger.debug(dataset[:]['bbx'])
        self.assertEqual(annotation.shape[0], len(dataset[:]['cropped_bbox']))
        self.assertEqual(annotation.shape[0], len(dataset[:]['bbx']))
        self.assertEqual(annotation.shape[0], len(dataset[:]['class']))
    def test_binary(self):
        dataset = dataloader_binairy()
        first_data = dataset[1]
        logger.debug(first_data)
        plt.imshow(first_data[0].numpy(),cmap='gray')
        plt.show()

    def test_dataloader_length(self):

        dataset = dataloader_normal()
        first_data = dataset[0]
        logger.debug(first_data)
        #logger.debug(first_data[0].numpy().shape)
        #plt.imshow(first_data[0].numpy().transpose(1, 2, 0))
        #plt.show()

