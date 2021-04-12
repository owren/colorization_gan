import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from train import *


def create_generator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(128, use_bias=False, kernel_size=(3, 3), strides=(2, 2), padding="same",
                            input_shape=(150, 150, 1)))

    assert model.output_shape == (None, 75, 75, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.5, input_shape=(100,)))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, use_bias=False, kernel_size=(3, 3), padding="same", strides=(3, 3)))
    assert model.output_shape == (None, 25, 25, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(3, 3), padding="same", use_bias=False))
    assert model.output_shape == (None, 75, 75, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False,
                                     activation="tanh"))
    assert model.output_shape == (None, 150, 150, 3)

    return model


def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(32, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def main():
    ds = tf.keras.preprocessing.image_dataset_from_directory("data/seg_train/forest",
                                                             label_mode=None,
                                                             batch_size=BATCH_SIZE,
                                                             image_size=(150, 150))

    generator = create_generator()
    discriminator = create_discriminator()
    learning_rate = 2e-4
    g_optimizer = tf.keras.optimizers.Adam(learning_rate)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train(generator, discriminator, ds, cross_entropy, g_optimizer, d_optimizer)


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.run_functions_eagerly(True)

    noise_dim = 10
    image_width = 150
    image_height = 150
    BATCH_SIZE = 32

    main()
