import unittest
from matplotlib import pyplot as plt
import pandas as pd
from src.dataloader import dataloader_normal
import logging

logging.basicConfig(level=logging.DEBUG)

class TestDataLoader(unittest.TestCase):

    def test_binary(self):
        pass 
    
    def test_length_annotation(self):
        
        annotation = pd.read_csv("C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv")
        dataset = dataloader_normal("C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv", "C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/images")
        
        logging.debug(dataset[0])  # Utiliser logging.debug au lieu de 35.debug

        plt.imshow(dataset.data["cropped_bbox"][0])
        plt.show()

        self.assertEqual(annotation.shape[0], len(dataset.data["cropped_bbox"]))
        self.assertEqual(annotation.shape[0], len(dataset.data["bbox"]))
        self.assertEqual(annotation.shape[0], len(dataset.data["label"]))

    def test_dataloader_length(self):
        
        annotation = pd.read_csv("data/train/_annotations.csv")
        dataset = dataloader_normal("data/train/_annotations.csv", "data/train/images")
        first_data = dataset[0]
        
        logging.debug(first_data)
        logging.debug(first_data[0].numpy().shape)
        
        plt.imshow(first_data[0].numpy().transpose(1, 2, 0))
        plt.show()

        self.assertEqual(annotation.shape[0], len(dataset.data["cropped_bbox"]))
        
    def test_binary(self):
        pass     

if __name__ == '__main__':
    unittest.main() 


    
    
   