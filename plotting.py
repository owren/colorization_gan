import os
import random
from config import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def print_loss(losses):
    np_arr = np.array(losses)
    losses_sum = np.sum(np_arr, axis=0)

    print("Gen total loss: " + str(losses_sum[0]))
    print("Gen loss: " + str(losses_sum[1]))
    print("L1 loss: " + str(losses_sum[2]))

    print("Disc total loss: " + str(losses_sum[3]))
    print("Disc Gen loss: " + str(losses_sum[4]))
    print("Disc Real loss: " + str(losses_sum[5]))


def load_one_img():
    filename = random.choice(os.listdir("data/seg_train/forest/sub"))
    path = "data/seg_train/forest/sub/" + filename

    rgb_image = tf.keras.preprocessing.image.load_img(path, target_size=(HEIGHT, WIDTH))
    rgb_image_tensor = tf.keras.preprocessing.image.img_to_array(rgb_image)
    rgb_image_tensor = tf.expand_dims(rgb_image_tensor, axis=0)

    return rgb_image_tensor


def plot_one(epoch, generator):
    """Plot a real image and a generated image from the grayscale version.

    Args:
        epoch: Integer inidicating the current epoch.
        generator: The keras generator model.
    """

    rgb_image_tensor = load_one_img()
    rgb_image_tensor /= 255.

    yuv_img = tf.image.rgb_to_yuv(rgb_image_tensor)
    y_channel = yuv_img[..., :1]

    uv_channel = generator(y_channel, training=False)
    uv_channel /= 2
    numpy_check = uv_channel.numpy()

    yuv_from_gen = tf.concat([y_channel, uv_channel], axis=3)
    rgb_from_gen = tf.image.yuv_to_rgb(yuv_from_gen)

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
