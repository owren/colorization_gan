import os
import random
from config import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv


def print_loss(losses):
    np_arr = np.array(losses)
    losses_sum = np.mean(np_arr, axis=0)

    wtr = csv.writer(open("losses.csv", "a"), delimiter=',', lineterminator='\n')
    wtr.writerow([losses_sum])


def load_one_img(ds):

    for img in ds.take(1):
        img = img[1, ...]
        yuv_image_tensor = tf.expand_dims(img, axis=0)

        return yuv_image_tensor


def plot_one(epoch, ds, discriminator, generator):
    """Plot a real image and a generated image from the grayscale version.

    Args:
        epoch: Integer inidicating the current epoch.
        discriminator: The keras discriminator model.
        ds: Dataset.
        generator: The keras generator model.
    """

    yuv_image_tensor = load_one_img(ds)

    y_channel = yuv_image_tensor[..., :1]
    uv_channel = generator(y_channel, training=False)
    uv_channel /= 2

    yuv_from_gen = tf.concat([y_channel, uv_channel], axis=3)
    rgb_from_gen = tf.image.yuv_to_rgb(yuv_from_gen)

    rgb_image_tensor = tf.image.yuv_to_rgb(tf.concat([y_channel, yuv_image_tensor[..., 1:] / 2], axis=3))
    images = [rgb_image_tensor[0, ...], rgb_from_gen[0, ...]]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.imshow(images[i])

    plt.text(-45, -45, "Epoch: " + str(epoch), fontsize=18)
    plt.show()

    mean = tf.math.reduce_mean(uv_channel[0, ...])
    std = tf.math.reduce_std(uv_channel[0, ...])

    #print("mean: " + str(mean))
    #print("std: " + str(std))
