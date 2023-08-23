import unittest
from matplotlib import pyplot as plt
import pandas as pd
from src.dataloader import dataloader_normal, dataloader_binairy
from src.utils.setup_logger import logger


class TestDataLoader(unittest.TestCase):

    def test_length_annotation(self):
        annotation = pd.read_csv("data/train/_annotations.csv")
        dataset = dataloader_normal()
        self.assertEqual(annotation.shape[0], len(dataset.data['cropped_bbox']))
        self.assertEqual(annotation.shape[0], len(dataset.data['bbox']))
        self.assertEqual(annotation.shape[0], len(dataset.data['label']))

    def test_binary(self):
        # TODO: Apply same test for binary image and Apply test if it's a binary image or not
        dataset = dataloader_binairy()
        first_data = dataset[1]
        logger.debug(first_data)
        plt.imshow(first_data[0].numpy(), cmap='gray')
        plt.show()
