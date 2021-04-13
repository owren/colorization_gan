import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from train import *


def downsample(filters, kernel_size, strides, batchnorm=True):
    layer = tf.keras.Sequential()
    layer.add(layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    layer.add(tf.keras.layers.LeakyReLU())

    return layer


def upsample(filters, kernel_size, strides, dropout):
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding="same",
                                              use_bias=False))
    layer.add(tf.keras.layers.BatchNormalization())

    if dropout:
        layer.add(tf.keras.layers.Dropout(0.5))

    layer.add(tf.keras.layers.LeakyReLU())

    return layer


def new_generator():
    inp = tf.keras.layers.Input(shape=[128, 128, 1])

    down_stack = [
        downsample(64, (4, 4), (2, 2), batchnorm=False),
        downsample(128, (4, 4), (2, 2)),
        downsample(256, (4, 4), (2, 2)),
        downsample(256, (4, 4), (2, 2))
    ]

    up_stack = [
        upsample(256, (4, 4), (2, 2), True),
        upsample(128, (4, 4), (2, 2), False),
        upsample(64, (4, 4), (2, 2), False)
    ]

    output = tf.keras.layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding="same", activation="tanh")

    x = inp
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = output(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    #tf.keras.utils.plot_model(model, to_file="gen.png", show_shapes=True, dpi=64)

    return model


def new_discriminator():

    inp = tf.keras.layers.Input(shape=[128, 128, 1], name="input_image")
    tar = tf.keras.layers.Input(shape=[128, 128, 3], name="target_image")

    input_layer = tf.keras.layers.concatenate([inp, tar])

    c1 = downsample(32, (4, 4), (2, 2))(input_layer)
    c2 = downsample(64, (4, 4), (2, 2))(c1)
    c3 = downsample(128, (4, 4), (2, 2))(c2)

    zero_pad_1 = tf.keras.layers.ZeroPadding2D()(c3)
    c4 = downsample(256, (4, 4), (1, 1))(zero_pad_1)

    zero_pad_2 = tf.keras.layers.ZeroPadding2D()(c4)

    output = tf.keras.layers.Conv2D(1, kernel_size=(4, 4), strides=(1, 1))(zero_pad_2)

    model = tf.keras.Model(inputs=[inp, tar], outputs=output)

    #tf.keras.utils.plot_model(model, to_file="disc.png", show_shapes=True, dpi=64)

    return model


def normalize(img):

  img = (img / 127.5) - 1

  return img

def main():
    ds = tf.keras.preprocessing.image_dataset_from_directory("data/seg_train/forest",
                                                             label_mode=None,
                                                             batch_size=BATCH_SIZE,
                                                             image_size=(128, 128))


    ds = ds.map(normalize)
    generator = new_generator()
    discriminator = new_discriminator()
    learning_rate = 2e-4
    g_optimizer = tf.keras.optimizers.Adam(learning_rate)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train(generator, discriminator, ds, cross_entropy, g_optimizer, d_optimizer)


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.run_functions_eagerly(True)

    image_width = 128
    image_height = 128
    BATCH_SIZE = 32

    main()
