import unittest

from src.dataloader import dataloader_normal
from src.utils.setup_logger import logger


class TestDataLoader(unittest.TestCase):

    def test_binary(self):
        pass

    def test_dataloader_length(self):

        dataset = dataloader_normal()
        first_data = dataset[0]
        logger.debug(first_data)
        print(dataset[0][0].size())
