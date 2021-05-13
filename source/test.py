import numpy as np
import tensorflow as tf
import os

from source.config import TEST_PATH, BATCH_SIZE, DATA_PATH, RESULT_PATH, MODEL_PATH
from source.data_loader import load_data
from source.train import generator_loss, discriminator_loss
from source.utility import get_channels


def store_image(generated_uv, y, uv):
    """Store a sample batch which compares the real and generated images in the 'result' folder.

    Args:
        generated_uv: The generated uv channel of the y channel.
        y: The real grayscale part of the image.
        uv: The real UV part of the image.
    """
    generated_uv /= 2
    uv /= 2

    generated_images = tf.concat([y, generated_uv], axis=3)
    real_images = tf.concat([y, uv], axis=3)

    rgb_generated = tf.image.yuv_to_rgb(generated_images)
    rgb_real = tf.image.yuv_to_rgb(real_images)

    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    for i in range(BATCH_SIZE):
        tf.keras.preprocessing.image.save_img(RESULT_PATH + "gen_img_" + str(i) + ".png", rgb_generated[i, ...])
        tf.keras.preprocessing.image.save_img(RESULT_PATH + "real_img_" + str(i) + ".png", rgb_real[i, ...])


def main():
    """Tests a trained model on a test dataset.

    Iterate through the dataset and calculate the mean loss of the model.
    The first batch of the dataset serves as a sample used to compare generated images
    and the real images. The samples are saved in the 'result' folder.

    """

    # Get the latest model from the current experiment
    with open(MODEL_PATH + "/model.txt", "r") as f:
        gen_filename = f.readline().strip("\n")
        disc_filename = f.readline().strip("\n")

    if gen_filename == "" or disc_filename == "":
        raise EnvironmentError("No model trained for the current experiment")

    generator_trained = tf.keras.models.load_model(MODEL_PATH + gen_filename)
    discriminator_trained = tf.keras.models.load_model(MODEL_PATH + disc_filename)

    ds = load_data(DATA_PATH)

    gen_loss = []
    disc_loss = []

    obtain_sample = True

    for image_batch in ds:
        y, uv, edge = get_channels(image_batch)
        generated_image = generator_trained([y, edge], training=False)

        if obtain_sample:
            store_image(generated_image, y, uv)
            obtain_sample = False

        disc_real_output = discriminator_trained([y, uv], training=False)
        disc_gen_output = discriminator_trained([y, generated_image], training=False)

        gen_loss.append(generator_loss(disc_gen_output, generated_image, uv))
        disc_loss.append(discriminator_loss(disc_real_output, disc_gen_output))

    gen_loss = np.array(gen_loss)
    disc_loss = np.array(disc_loss)

    print("Generator Test Loss: " + str(np.mean(gen_loss, axis=0)[0]))
    print("Discriminator Test Loss: " + str(np.mean(disc_loss, axis=0)[0]))


if __name__ == "__main__":
    main()
