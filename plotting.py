import os
import random
from config import *
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_one(epoch, generator):
    """Plot a real image and a generated image from the grayscale version.

    Args:
        epoch: Integer inidicating the current epoch.
        generator: The keras generator model.
    """
    filename = random.choice(os.listdir("data/seg_train/forest/sub"))
    path = "data/seg_train/forest/sub/" + filename

    rgb_image = tf.keras.preprocessing.image.load_img(path, target_size=(HEIGHT, WIDTH))
    rgb_image_tensor = tf.keras.preprocessing.image.img_to_array(rgb_image)
    rgb_image_tensor = tf.expand_dims(rgb_image_tensor, axis=0)
    grayscale_image = tf.image.rgb_to_grayscale(rgb_image_tensor)
    grayscale_image_normalized = grayscale_image / 255.

    gen_image = generator(grayscale_image_normalized, training=False) / 2
    gen_image = tf.concat([grayscale_image_normalized, gen_image], axis=3)
    gen_image = tf.image.yuv_to_rgb(gen_image)
    images = [rgb_image, gen_image[0, ...]]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.imshow(images[i])

    plt.text(-45, -45, "Epoch: " + str(epoch), fontsize=18)
    plt.show()
