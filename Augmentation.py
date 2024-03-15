import os
import numpy as np
from PIL import Image
import cv2
from skimage import io
from keras.preprocessing.image import ImageDataGenerator

def augmentation_data(input_dir, size, augmentation_params=None, num_augmented_images=240):
    if augmentation_params is None:
        augmentation_params = {
            'featurewise_center': True,
            'rotation_range': (-10, 10),
            'width_shift_range': 0.03,
            'height_shift_range': 0.03,
            'zoom_range': 0.02
        }

    datagen = ImageDataGenerator(**augmentation_params)

    dataset = []
    myimages = os.listdir(input_dir)
    myimages.sort()

    for i, image_name in enumerate(myimages):
        if image_name.split('.')[1] == 'png':
            image = io.imread(os.path.join(input_dir, image_name))
            image = Image.fromarray(image, "L")
            image = image.resize((size, size))
            dataset.append(np.array(image))

    x = np.array(dataset)
    x = cv2.merge([x, x, x])

    i = 0
    for batch in datagen.flow(x, batch_size=1, 
                              save_to_dir=input_dir, 
                              save_prefix='augmented_', 
                              save_format='png'):
        i += 1
        if i > num_augmented_images:
            break

# Example usage:
input_dir = '/content/drive/MyDrive/multitemporell/T10/'

augment_data(input_dir, output_dir)