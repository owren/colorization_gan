import tensorflow as tf
from config import ENABLE_CUDA, DEBUG_MODE, loss_filename, DATA_PATH, VALIDATION_PATH
from data_loader import load_data
from discriminator import create_discriminator
from generator import create_generator
from train import train


def main():
    """Creates the dataset, generator, and discrminiator then begin the training process"""

    train_ds = load_data(DATA_PATH)
    validation_ds = load_data(VALIDATION_PATH)

    generator = create_generator()
    discriminator = create_discriminator()

    # Clear the loss filename in case experiment runs with same experiment name
    open(loss_filename, "w").close()

    train(generator, discriminator, train_ds, validation_ds)


if __name__ == "__main__":

    # Only neccessary if CUDA is enabled
    if ENABLE_CUDA:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Enable when debugging '@tf.function'
    if DEBUG_MODE:
        tf.config.run_functions_eagerly(True)

    main()
