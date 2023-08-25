import unittest
import logging
from dataloader_normal import dataloader_normal  

logging.basicConfig(level=logging.DEBUG)

class TestDataLoader(unittest.TestCase):

    def test_binary(self):
        pass 
    
    def test_length_annotation(self):
        data_file = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv'  
        images_folder = ':/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data'  

        dataset = dataloader_normal(data_file, images_folder)
        data_length = len(dataset)

        annotations = dataset.load_data(data_file)['images']
        annotations_length = len(annotations)

        logging.debug(f"Data length: {data_length}, Annotations length: {annotations_length}")
        self.assertEqual(data_length, annotations_length)

    def test_dataloader_length(self):
        
        data_file = 'C:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data/_annotations.csv'  
        images_folder = 'p:/Users/ACER/Desktop/Stage/code/arabian-calligraphy-demo/Data'  

        dataset = dataloader_normal(data_file, images_folder)

        expected_length = len(dataset.load_data(data_file)['images'])
        actual_length = len(dataset)

        logging.debug(f"Expected length: {expected_length}, Actual length: {actual_length}")
        self.assertEqual(actual_length, expected_length)

        pass