import cv2
from PIL import Image
import tensorflow as tf
import matplotlib
import matplotlib.image as mplimg
import Metrics as M
from LoadData import data_loader
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os 

data_dir='./Dataset/'
dir_save='./outputs/'
checkpoint_model_file = '/weights'



def save_img(l,moved, moving, static,size):
    nb, nh, nw, nc = moving.shape
    nb, nh, nw, nc = static.shape
    # moved = model({'moving': moving, 'static': static}, training=True)
    moving = tf.image.resize(moving, size)
    static = tf.image.resize(static, size)
    moved = tf.image.resize(moved,size)
    #Convert the tensors to 8-bit images.
    moved = moved.numpy().squeeze(axis=-1) * 255.0
    moved = moved.astype(np.uint8)
    moving = moving.numpy().squeeze(axis=-1) * 255.0
    moving = moving.astype(np.uint8)
    static = static.numpy().squeeze(axis=-1) * 255.0
    static = static.astype(np.uint8)

    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    cv2.imwrite(os.path.join(dir_save, str(l) + ".png"), moved)
    print("\n")


def main(args):
    model = tf.keras.models.load_model(checkpoint_model_file1)
    model.load_weights(checkpoint_model_file)
    print('| loading model file %s... ' % checkpoint_model_file
    )
    data_loader = DataLoader(data_dir)
    x_train, x_test, y_train , y_test  =data_loader.preprocess_data()
    test_dataset = Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

    criterion = M.mse_loss
    m_test = tf.keras.metrics.Mean(name='Dicescore')
    for i, (static, moving) in enumerate(test_dataset):
        m_test.reset_states()
        start_time = time.time()  # Start measuring time
        nb, nh, nw, nc = tf.keras.backend.int_shape(moving)  # moving.shape
        moved = model({'moving': moving, 'static': static}, training=False)
        loss_test = criterion(moved, static)
        m_test.update_state(loss_test)
        print('Test iteration {}, Avg. Dice score: {:5.4f}'.format(i, m_test.result()))
        diff_time = time.time() - start_time
        print("Time taken for image {}: {:.4f} seconds".format(i, diff_time))

        # Calculate additional metrics
        psnr = M.psnr(moved, static)
        dice= M.Dicescore(moved, static)
        print('PSNR: {:.4f}, dice: {:.4f}'.format(psnr, dice))

        #save moved image
        save_img(i,moved, moving, static,args.size)

if __name__ == "__main__":
    # Assuming you pass arguments using argparse
    import argparse

    parser = argparse.ArgumentParser(description="Affine Registration Testing")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument('--size', type=int, default=640)  
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    args = parser.parse_args()

    main(args)

