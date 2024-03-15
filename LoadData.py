import os
import numpy as np
import tensorflow as tf
from PIL import Image

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

    def load_data(self):
        x_data, y_data = [], []
        for file_name in sorted(os.listdir(os.path.join(self.data_dir, 'static'))):
            x_data.append(os.path.join(self.data_dir, 'static', file_name))
        for file_name in sorted(os.listdir(os.path.join(self.data_dir, 'moving'))):
            y_data.append(os.path.join(self.data_dir, 'moving', file_name))

        # Assuming 80% for training and 20% for testing
        split_index = int(0.8 * len(x_data))

        x_train, x_test = np.array(x_data[:split_index]), np.array(y_data[:split_index])
        y_train, y_test = np.array(x_data[split_index:]), np.array(y_data[split_index:])

        return x_train, y_train, x_test, y_test

    def preprocess_images(self, images):
        images = images.astype(np.float32) / 255.0
        images = images[..., None]
        images = tf.image.resize(images, (256, 256))
        return images

    def preprocess_data(self):
        x_train = self.preprocess_images(np.array([np.array(Image.open(fname)) for fname in self.x_train]))
        y_train = self.preprocess_images(np.array([np.array(Image.open(fname)) for fname in self.y_train]))
        x_test = self.preprocess_images(np.array([np.array(Image.open(fname)) for fname in self.x_test]))
        y_test = self.preprocess_images(np.array([np.array(Image.open(fname)) for fname in self.y_test]))
        return x_train, x_test, y_train , y_test

data_loader = DataLoader('/content/drive/MyDrive/dataa/')
x_train, x_test, y_train , y_test  =data_loader.preprocess_data()