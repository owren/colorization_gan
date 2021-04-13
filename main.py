import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from train import *


def create_generator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(256, use_bias=False, kernel_size=(4, 4), strides=(2, 2), padding="same",
                            input_shape=(152, 152, 1)))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.5))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.5))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False,
                                     activation="tanh"))
    print(model.output_shape)
    assert model.output_shape == (None, 152, 152, 3)

    return model


def new_discriminator():
    model = tf.keras.Sequential()

    inp = tf.keras.layers.Input(shape=[152, 152, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[152, 152, 3], name='target_image')

    model.add(tf.keras.layers.concatenate([inp, tar]))
    model.add(layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.ZeroPadding2D())

    model.add(layers.Conv2D(1, kernel_size=(4, 4), strides=(1, 1)))

    return model


def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding="same", input_shape=(152, 150, 3)))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())

    model.add(layers.Dense(2048))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def main():
    ds = tf.keras.preprocessing.image_dataset_from_directory("data/seg_train/forest",
                                                             label_mode=None,
                                                             batch_size=BATCH_SIZE,
                                                             image_size=(152, 152))

    generator = create_generator()
    discriminator = new_discriminator()
    learning_rate = 2e-4
    g_optimizer = tf.keras.optimizers.Adam(learning_rate)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train(generator, discriminator, ds, cross_entropy, g_optimizer, d_optimizer)


if __name__ == "__main__":
    #physical_devices = tf.config.list_physical_devices("GPU")
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.run_functions_eagerly(True)

    image_width = 150
    image_height = 150
    BATCH_SIZE = 32

    main()
