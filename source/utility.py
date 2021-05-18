import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


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

    wtr = csv.writer(open(filename, 'a'), delimiter=',', lineterminator='\n')
    wtr.writerow(losses_sum)


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


def plot_loss(path_to_loss, file_name):
    """Plots loss to png file in lossfig directory.

    Creates a graph of each loss and all the losses together
    Args:
        path_to_loss: string, path to csv file with six different losses
        file_name: string, name of the csv file with six different losses
    """
    path = '../'
    labels = ['gen_total_loss', 'gen_loss', 'l1_loss', 'disc_total_loss', 'disc_gen_loss', 'disc_real_loss']
    with open(path_to_loss + file_name + '.csv', newline='') as f:
        reader = csv.reader(f)
        data = np.array(list(reader))

    try:
        os.mkdir(path + 'lossfig/losses_' + file_name)
    except:
        pass

    epoch_count = range(1, data.shape[0] + 1)
    for i in range(data.shape[1]):
        plt.figure()
        plt.plot(epoch_count, data[:, i].astype('float32'))
        plt.xlabel('Epoch')
        plt.ylabel(labels[i])
        plt.savefig(path + 'lossfig/losses_' + file_name + '/' + labels[i] + '.png')

    plt.figure()
    for i in range(data.shape[1]):
        plt.plot(epoch_count, data[:, i].astype('float32'))
    plt.legend(labels)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(path + 'lossfig/losses_' + file_name + '/all_loss.png')


def plot_train_val_loss(path_to_file, train_filename, val_filename):
    """"Plots the train loss with the validation loss

    Args:
        path_to_file: string, to the loss files
        train_filename: string, filename for train loss
        val_filename: string, filename for validation loss
    """
    path = '../'
    labels = ['gen_total_loss', 'gen_loss', 'l1_loss', 'disc_total_loss', 'disc_gen_loss', 'disc_real_loss']
    with open(path_to_file + train_filename + '.csv', newline='') as f:
        reader = csv.reader(f)
        train_data = np.array(list(reader))
    with open(path_to_file + val_filename + '.csv', newline='') as f:
        reader = csv.reader(f)
        val_data = np.array(list(reader))

    if train_data.shape == val_data.shape:
        # change label number for the type of loss that should be plotted
        label = 0
        print(train_data.shape, val_data.shape)
        epoch_count = range(1, train_data.shape[0] + 1)
        plt.figure()
        plt.plot(epoch_count, val_data[:len(epoch_count), label].astype('float32'))
        plt.plot(epoch_count, train_data[:len(epoch_count), label].astype('float32'))
        plt.legend(['validation', 'train'])
        plt.xlabel('Epoch')
        plt.ylabel(labels[label])
        plt.show()
