import tensorflow as tf
from discriminator import create_discriminator
from generator import create_generator
from train import *
import cv2


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


def load_data(path):
    _, _, filenames = next(os.walk(path))

    #shuffle = random.shuffle(filenames)
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

        image /= 255.
        image = tf.image.rgb_to_yuv(image)
        y = image[..., :1]
        uv_normalize = image[..., 1:] * 2

        img_numpy = y.numpy()
        img_numpy = np.array(img_numpy * 255, dtype=np.uint8)
        edge = cv2.Laplacian(img_numpy, HEIGHT, WIDTH)
        edge = tf.cast(edge, tf.float32)
        edge_tensor = tf.convert_to_tensor(edge)
        edge_tensor = tf.expand_dims(edge_tensor, axis=2)

        image = tf.concat([y, uv_normalize, edge_tensor], axis=2)
        ds.append(image)

    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = tf.data.Dataset.shuffle(ds, buffer_size=500)
    ds = ds.batch(BATCH_SIZE)


    return ds



def main():
    """Creates the dataset, generator, and discrminiator then begin the training process"""

    ds = load_data("data/seg_train/forest/sub")


    '''
    ds = tf.keras.preprocessing.image_dataset_from_directory("data/seg_train/forest",
                                                             label_mode=None,
                                                             batch_size=BATCH_SIZE,
                                                             color_mode="rgb",
                                                             image_size=(150, 150),
                                                             shuffle=True)
    ds = ds.map(yuv_cast)
    
    # Randomly crop every image in the dataset
    ds = ds.map(lambda x: tf.map_fn(lambda y: tf.image.random_crop(y, size=(HEIGHT, WIDTH, 3)), x))
    
    for img in ds.take(1):
        img = img[1, ...]
        #img = tf.expand_dims(img, axis=0)
        grayscale_tensor = img[..., :1]
        #edge = tf.image.sobel_edges(grayscale_tensor)

        grayscale_numpy = grayscale_tensor.numpy()

        grayscale_numpy = np.array(grayscale_numpy * 255, dtype=np.uint8)
        plt.imshow(grayscale_numpy, cmap="gray")
        plt.show()

        #grayscale_cv = cv2.Canny(grayscale_numpy, HEIGHT, WIDTH)

        edge = cv2.Laplacian(grayscale_numpy, HEIGHT, WIDTH)
        plt.imshow(edge, cmap="gray")
        plt.show()
        print("x")
    '''

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
    #physical_devices = tf.config.list_physical_devices("GPU")
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Enable when debugging '@tf.function'.
    tf.config.run_functions_eagerly(True)

    main()
