import tensorflow as tf
from discriminator import create_discriminator
from generator import create_generator
from train import *


def yuv_cast(img):
    """Normalizes the RGB image and casts it to YUV.

    Args:
        img: An RGB image with the ranges (0, 255)

    Returns:
        An YUV image with range Y: (0, 1) and UV: (-1, 1)
    """
    img /= 255.
    img = tf.image.rgb_to_yuv(img)
    y = img[..., :1]
    uv_normalize = img[..., 1:] * 2
    img = tf.concat([y, uv_normalize], axis=3)

    return img


def main():
    """Creates the dataset, generator, and discrminiator then begin the training process"""

    ds = tf.keras.preprocessing.image_dataset_from_directory("data/seg_train/forest",
                                                             label_mode=None,
                                                             batch_size=BATCH_SIZE,
                                                             color_mode="rgb",
                                                             image_size=(HEIGHT, WIDTH))
    ds.shuffle(100)

    ds = ds.map(yuv_cast)

    generator = create_generator()
    discriminator = create_discriminator()

    train(generator, discriminator, ds)


if __name__ == "__main__":

    # Only neccessary if CUDA is enabled.
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Enable when debugging '@tf.function'.
    tf.config.run_functions_eagerly(True)

    main()
