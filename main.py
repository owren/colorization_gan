import tensorflow as tf
import numpy as np
from model import create_generator, create_discriminator
from train import *


def normalize(img):
    img /= 255.
    return tf.image.rgb_to_yuv(img)


def main():
    ds = tf.keras.preprocessing.image_dataset_from_directory("data/seg_train/forest",
                                                             label_mode=None,
                                                             batch_size=BATCH_SIZE,
                                                             color_mode="rgb",
                                                             image_size=(HEIGHT, WIDTH))
    ds.shuffle(100)
    ds = ds.map(normalize)

    generator = create_generator()
    discriminator = create_discriminator()

    train(generator, discriminator, ds)


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.run_functions_eagerly(True)

    main()
