import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv


def get_channels(batch):
    """Splits the batch to the Y, UV, and Edge parts of the batch.

    Args:
        batch: A batch from the dataset.

    Returns:
        y: The Y part of the image.
        uv: The UV part of the image.
        edge: The Edge part of the image.
    """
    y = batch[..., :1]
    uv = batch[..., 1:3]
    edge = batch[..., 3:]

    return y, uv, edge


def store_loss(losses, filename):
    """Store loss of the current session.

    The sequence of the losses is: gen_total_loss, gen_loss,
    l1_loss, disc_total_loss, disc_gen_loss, disc_real_loss

    Args:
        losses: The loss of the model for the current epoch.
        filename: CSV file where to store the losses.
    """
    np_arr = np.array(losses)
    losses_sum = np.mean(np_arr, axis=0)

    wtr = csv.writer(open(filename, "a"), delimiter=",", lineterminator="\n")
    wtr.writerow([losses_sum])


def load_one_img(ds):
    """Load one image from the dataset

    Args:
        ds: The tensorflow dataset.
    """
    for img in ds.take(1):
        img = img[1, ...]
        yuv_image_tensor = tf.expand_dims(img, axis=0)

        return yuv_image_tensor


def plot_one(epoch, ds, discriminator, generator):
    """Plot a real image and a generated image from the grayscale version.

    Args:
        epoch: Integer inidicating the current epoch.
        discriminator: The keras discriminator model.
        ds: Dataset.
        generator: The keras generator model.
    """
    yuv_image_tensor = load_one_img(ds)
    y_channel, real_uv_channel, edge_channel = get_channels(yuv_image_tensor)

    gen_uv_channel = generator([y_channel, edge_channel], training=False)

    disc_gen_result = discriminator([y_channel, gen_uv_channel], training=False)
    disc_gen_result = round(tf.math.reduce_mean(disc_gen_result).numpy(), 2)

    disc_real_result = discriminator([y_channel, real_uv_channel], training=False)
    disc_real_result = round(tf.math.reduce_mean(disc_real_result).numpy(), 2)

    gen_uv_channel /= 2

    yuv_from_gen = tf.concat([y_channel, gen_uv_channel], axis=3)
    rgb_from_gen = tf.image.yuv_to_rgb(yuv_from_gen)

    rgb_image_tensor = tf.image.yuv_to_rgb(tf.concat([y_channel, real_uv_channel / 2], axis=3))
    images = [rgb_image_tensor[0, ...], rgb_from_gen[0, ...]]
    results = [disc_real_result, disc_gen_result]

    fig = plt.figure()
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1)
        plt.axis("off")
        plt.text(55, -5, results[i], size=16)
        plt.imshow(images[i])

    plt.text(-45, -45, "Epoch: " + str(epoch), fontsize=18)
    plt.show()