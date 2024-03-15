import tensorflow as tf
from keras import backend as K
import math
"""
Args:
    y_pred (numpy.ndarray): Predicted matrix
    y_true (numpy.ndarray): Ground truth matrix
"""
def mse_loss(y_true, y_pred):

    # Calculate the squared difference between predicted and ground truth matrix
    loss= tf.square(y_pred - y_true)

    # Calculate the mean of the squared differences
    MSE_loss = tf.reduce_mean(loss)  # shape ()

    return MSE_loss


def PSNR(y_true, y_pred):

    # Assuming the maximum pixel value is 1.0
    max_pixel = 1.0

    # Calculate the Mean Squared Error (MSE) between y_true and y_pred
    mse= K.mean(K.square(y_pred - y_true))

    # Calculate the Peak Signal-to-Noise Ratio (PSNR) using the MSE
    PSNR= 10.0 * math.log10((max_pixel ** 2) / (mse)) 

    return PSNR


def dice_coef(y_true, y_pred, smooth=1):
    """
    Args:
    smooth (float): Smoothing factor to avoid division by zero (default: 1e-5)

    Returns:
    float: Dice coefficient between the predicted and ground truth image
    """
    # Calculate the intersection between the predicted and ground truth masks
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])

    # Calculate the Dice coefficient
    dice= (2. * intersection + smooth) / (K.sum(K.square(y_true),axis=[1,2]) + K.sum(K.square(y_pred),axis=[1,2]) + smooth)
    return dice

def ncc_loss(y_true, y_pred):

    eps = tf.constant(1e-9, 'float32')

    y_true_mean = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    # shape (N, 1, 1, C)

    # Calculate standard deviation
    static_std = tf.math.reduce_std(y_true, axis=[1, 2], keepdims=True)
    y_pred_std = tf.math.reduce_std(y_pred, axis=[1, 2], keepdims=True)
    # shape (N, 1, 1, C)

     # Normalize
    y_true_hat = (y_true - y_true_mean)/(y_true_std + eps)
    y_pred_hat = (y_pred - y_pred_mean)/(y_pred_std + eps)
    # shape (N, H, W, C)

    # Calculate normalized cross-correlation (NCC)
    ncc = tf.reduce_mean(y_true_hat * y_pred_hat)  # shape ()

    # Define loss as negative NCC
    loss = -ncc
    return loss