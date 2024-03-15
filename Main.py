import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
from numpy import asarray
from Augmentation import augmentation_data
import Metrics as M
import models as model
from LoadData import data_loader
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

data_dir= '../../dataset/'
model_dir = '../../weights/'

#train the models

def train_step(model,static,moving, criterion, optimizer):
    nb, nh, nw, nc = tf.keras.backend.int_shape(moving)
    with tf.GradientTape() as tape:
        moved = model({'moving': moving, 'static': static})
        loss_train = criterion(moved,static)
    grads = tape.gradient(loss_train, model.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_train



def test_step(model,static,moving, criterion):
    nb, nh, nw, nc = tf.keras.backend.int_shape(moving)
    moved = model({'moving': moving, 'static': static},, training=False)
    loss_test = criterion(moved,static)
    return loss_test


def main(args):
    data_loader = DataLoader(data_dir)
    x_train, x_test, y_train , y_test  =data_loader.preprocess_data()
#----------------------------------------------------------
#---------------------------------------------------
    from_tensor_slices = tf.data.Dataset.from_tensor_slices
    x_train = from_tensor_slices(x_train).shuffle(10).batch(args.batch_size)
    x_test = from_tensor_slices(x_test).shuffle(10).batch(args.atch_size)
    y_train = from_tensor_slices(y_train).shuffle(10).batch(args.batch_size)
    y_test = from_tensor_slices(y_test).shuffle(10).batch(args.batch_size)

    # Create a model instance.
    model = model.model(arg.input_shape)   # (input_shape=(256, 256))

    # Select optimizer and loss function.
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    criterion= M.mse_loss # or ncc_loss

    m_train = tf.keras.metrics.Mean(name='loss_train')
    m_test = tf.keras.metrics.Mean(name='loss_test')

    start_t = t = time.time()

    # Train and evaluate the model.
    for epoch in range(args.epochs):
        m_train.reset_states()
        m_test.reset_states()
        for i, (static,moving) in enumerate(zip(x_train,y_train)):
            loss_train = train_step(model, moving, static, criterion,
                                    optimizer)
            m_train.update_state(loss_train)
            cur_t = time.time()
            print("Total time: ", cur_t - start_t, "seconds")
            print('| - Epoch: %3d/%d\tTrain final Loss: %.6f'
              % (epoch + 1, args.epochs, m_train.result()))
        for i, (static,moving) in enumerate(zip(x_test,y_test)):
            loss_test = test_step(model, moving, static, criterion)
            m_test.update_state(loss_train)
            cur_t = time.time()
            print("Total time: ", cur_t - start_t, "seconds")
            print('| - Epoch: %3d/%d\tTest final Loss: %.6f'
              % (epoch + 1, args.epochs, m_test.result()))

        content = '| epo:%s/%s  train_loss_avg:%.4f test_loss_avg:%.4f ' \
                  % (epoch + 1, epochs, m_train.result(), m_test.result())

        print(content)
        with open(log_file, 'a') as appender:
          appender.write(content)
          
        # save check point model
        print("| saving check point model file... ", end="")
        if args.save_model:
          model.save_weights(checkpoint_model_file)
          print("done!")
        print("\n")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Cnn with Keras')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model or not')
    
    args = parser.parse_args()


    checkpoint_model_file = os.path.join(args.model_dir, "final.h5")
    log_file = os.path.join(args.model_dir, "log.txt")

    print("| training Cnn_model with Keras")
    print("| model will be saved in: %s" % model_dir)
    print("| log will be saved in: %s" % log_file)

    main(args)






















