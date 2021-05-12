import tensorflow as tf
from discriminator import create_discriminator
from generator import create_generator
from train import *
import cv2


def grayscale_to_edge_2(grayscale):
    grey_im = grayscale.numpy()

    du_dy = np.zeros(grey_im.shape)
    du_dy[1:-1, 1:-1] = grey_im[2:, 1:-1] - grey_im[1:-1, 1:-1]

    du_dx = np.zeros(grey_im.shape)
    du_dx[1:-1, 1:-1] = grey_im[1:-1, 2:] - grey_im[1:-1, 1:-1]

    grad_norm = np.sqrt(du_dx ** 2 + du_dy ** 2)

    edge = tf.cast(grad_norm, tf.float32)
    edge_tensor = tf.convert_to_tensor(edge)
    edge_tensor = tf.expand_dims(edge_tensor, axis=2)

    return grad_norm


def grayscale_to_edge(grayscale):
    img_numpy = grayscale.numpy()
    img_numpy = np.array(img_numpy * 255, dtype=np.uint8)
    edge = cv2.Laplacian(img_numpy, HEIGHT, WIDTH)
    edge = tf.cast(edge, tf.float32)
    edge_tensor = tf.convert_to_tensor(edge)
    edge_tensor = tf.expand_dims(edge_tensor, axis=2)

    return edge_tensor

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


def load_data(path):
    _, _, filenames = next(os.walk(path))

    # shuffle = random.shuffle(filenames)
    const = tf.constant(filenames)

    ds = []

    for img in const:
        image_string = tf.io.read_file(path + "/" + img)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)

        try:
            image = tf.image.random_crop(image, size=(HEIGHT, WIDTH, 3))
        except:
            image = tf.image.resize(image, size=(HEIGHT, WIDTH))

        y, uv = yuv_cast(image)
        edge_tensor = grayscale_to_edge_2(y)

        image = tf.concat([y, uv, edge_tensor], axis=2)
        ds.append(image)

    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = tf.data.Dataset.shuffle(ds, buffer_size=500)
    ds = ds.batch(BATCH_SIZE)

    return ds


def main():
    """Creates the dataset, generator, and discrminiator then begin the training process"""

    ds = load_data("data/seg_train/forest/sub")

    generator = create_generator()
    discriminator = create_discriminator()

    checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                     discriminator_optimizer=d_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Restore latest checkpoint (not sure if works)
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    train(generator, discriminator, ds, checkpoint)


if __name__ == "__main__":
    # Only neccessary if CUDA is enabled.
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Enable when debugging '@tf.function'.
    tf.config.run_functions_eagerly(True)

    main()
