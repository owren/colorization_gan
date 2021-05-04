import tensorflow as tf
import time
from config import *
from plotting import *


def discriminator_loss(disc_real_output, disc_gen_output):
    """Calculate the loss of the discriminator.

    Args:
        disc_real_output: A tensor which represent the discriminator output when input the real image.
        disc_gen_output:A tensor which represent the discriminator output when input the generated image.

    Returns:
        The total loss of the discrminiator
    """
    gen_loss = cross_entropy(tf.zeros_like(disc_gen_output), disc_gen_output)
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    total_loss = real_loss + gen_loss

    return total_loss, gen_loss, real_loss


def generator_loss(disc_gen_output, generated_image, real_image):
    """Calculate the loss of the generator.

    Args:
        disc_gen_output: A tensor which represent the patch output of the discriminator.
        generated_image: A tensor which represent the generated image with only the UV channels.
        real_image: A tensor which represent the real image with only the UV channels.

    Returns:
        The total loss of the generator, the loss from the discriminator, and the
        rmse loss when comparing to the real image.
    """
    gen_loss = cross_entropy(tf.ones_like(disc_gen_output), disc_gen_output)

    l1_abs = tf.abs(real_image - generated_image)
    l1_loss = tf.reduce_mean(l1_abs)
    l1_loss *= 10

    gen_total_loss = gen_loss + l1_loss
    return gen_total_loss, gen_loss, l1_loss


@tf.function
def train_step(generator, discriminator, images):
    """Performs one train step on the GAN by using one batch from the dataset.

    Uses GradientTape to perform backpropagation one step each at the discriminiator and generator.

    Args:
        generator: A Keras Model which represent the Generator of the GAN.
        discriminator: A Keras Model which represent the Discrminiator of the GAN.
        images: A image batch from the tensorflow dataset.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Get the grayscale part of the images in the batch.
        grayscale_batch =  images[..., :1]

        # Get the UV part of the images in the batch.
        uv_batch = images[..., 1:]

        generated_image = generator(grayscale_batch, training=True)

        disc_real_output = discriminator([grayscale_batch, uv_batch], training=True)
        disc_gen_output = discriminator([grayscale_batch, generated_image], training=True)

        gen_total_loss, gen_loss, l1_loss = generator_loss(disc_gen_output, generated_image, uv_batch)
        disc_total_loss, disc_gen_loss, disc_real_loss = discriminator_loss(disc_real_output, disc_gen_output)

    gradients_of_generator = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_total_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_total_loss, gen_loss, l1_loss, disc_total_loss, disc_gen_loss, disc_real_loss


def train(generator, discriminator, dataset, checkpoint):
    """Begins the training process of the GAN.

    Itereates through the total number of epochs and apply a trainstep on the GAN for each
    batch in the dataset.

    Args:
        generator: A Keras Model which represent the Generator of the GAN.
        discriminator: A Keras Model which represent the Discrminiator of the GAN.
        dataset: A tensorflow dataset.
        checkpoint: Tensorflow checkpoint object used to store checkpoints of the progress.
    """
    plot_one(-1, dataset, discriminator, generator)
    for epoch in range(EPOCHS):
        losses = []
        start = time.time()
        for image_batch in dataset:
            loss = train_step(generator, discriminator, image_batch)
            losses.append(loss)

        print("Epoch " + str(epoch + 1) + ": " + str(round(time.time() - start, 3)) + " seconds")
        store_loss(losses)

        plot_one(epoch, dataset, discriminator, generator)

        # Save checkpoint every 20 epoch
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
