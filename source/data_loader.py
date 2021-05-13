import os
import numpy as np
import tensorflow as tf
from config import HEIGHT, WIDTH, BATCH_SIZE


def grayscale_to_edge(grayscale):
    """Casts a grayscale image to an edge image.

    Returns:
        The edge image with size HEIGHT * WIDTH * 1.
    """
    grey_im = grayscale.numpy()

    du_dy = np.zeros(grey_im.shape)
    du_dy[1:-1, 1:-1] = grey_im[2:, 1:-1] - grey_im[1:-1, 1:-1]

    du_dx = np.zeros(grey_im.shape)
    du_dx[1:-1, 1:-1] = grey_im[1:-1, 2:] - grey_im[1:-1, 1:-1]

    grad_norm = np.sqrt(du_dx ** 2 + du_dy ** 2)

    #edge = tf.cast(grad_norm, tf.float32)
    #edge_tensor = tf.convert_to_tensor(edge)
    #edge_tensor = tf.expand_dims(edge_tensor, axis=2)

    return grad_norm


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
    uv_normalized = img[..., 1:] * 2

    return y, uv_normalized


def load_data(ds_path):
    """Loads the data from ds_path and creates edge detection versions of the images.

    Returns:
        A tensorflow dataset where each record is an image with the size HEIGHT * WIDTH * 4.
        The first 3 values is the YUV values of the image, the last value is the edge
        detection values of the image.
    """
    _, _, filenames = next(os.walk(ds_path))
    const = tf.constant(filenames)

    ds = []

    for img in const:
        image_string = tf.io.read_file(ds_path + "/" + img)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)

        # Try except used to handle data which does not have the common size
        try:
            image = tf.image.random_crop(image, size=(HEIGHT, WIDTH, 3))
        except:
            image = tf.image.resize(image, size=(HEIGHT, WIDTH))

        y, uv = yuv_cast(image)
        edge_tensor = grayscale_to_edge(y)

        image = tf.concat([y, uv, edge_tensor], axis=2)
        ds.append(image)

    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = tf.data.Dataset.shuffle(ds, buffer_size=500)
    ds = ds.batch(BATCH_SIZE)

    return ds
