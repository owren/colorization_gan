import numpy as np
import tensorflow as tf
import time
from config import cross_entropy, g_optimizer, d_optimizer, EPOCHS, MODEL_SAVE, MODEL_PATH
from utility import get_channels, plot_one, store_loss


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

    Returns:
        The different losses from the generator and discriminator.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Get the Y part, UV part, and edge version of the images in the batch.
        y, uv, edge = get_channels(images)

        generated_image = generator([y, edge], training=True)

        disc_real_output = discriminator([y, uv], training=True)
        disc_gen_output = discriminator([y, generated_image], training=True)

        # Caluclate the loss of the generator and the discriminator.
        gen_total_loss, gen_loss, l1_loss = generator_loss(disc_gen_output, generated_image, uv)
        disc_total_loss, disc_gen_loss, disc_real_loss = discriminator_loss(disc_real_output, disc_gen_output)

    # Caluclates the gradeints of the generator and discrminiator with respect to the loss.
    gradients_of_generator = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_total_loss, discriminator.trainable_variables)

    # Apply backpropagation to the generator and discriminator given the gradients.
    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_total_loss, gen_loss, l1_loss, disc_total_loss, disc_gen_loss, disc_real_loss


def validate(generator, discriminator, validation_ds):
    """Validate inbetween epochs to check for overfitting.

    Args:
        generator: Keras model.
        discriminator: Keras model.
        validation_ds: Tensorflow dataset used for validation.
    """
    gen_loss = []
    disc_loss = []

    for image_batch in validation_ds:
        y, uv, edge = get_channels(image_batch)
        generated_image = generator([y, edge], training=False)

        disc_real_output = discriminator([y, uv], training=False)
        disc_gen_output = discriminator([y, generated_image], training=False)

        gen_loss.append(generator_loss(disc_gen_output, generated_image, uv))
        disc_loss.append(discriminator_loss(disc_real_output, disc_gen_output))

    gen_loss = np.array(gen_loss)
    disc_loss = np.array(disc_loss)

    print("Generator Validation Loss: " + str(np.mean(gen_loss, axis=0)[0]))
    print("Discriminator Validation Loss: " + str(np.mean(disc_loss, axis=0)[0]))


def train(generator, discriminator, train_ds, validation_ds):
    """Begins the training process of the GAN.

    Itereates through the total number of epochs and apply a trainstep on the GAN for each
    batch in the dataset.

    Args:
        generator: A Keras Model which represent the Generator of the GAN.
        discriminator: A Keras Model which represent the Discrminiator of the GAN.
        train_ds: Tensorflow dataset used for training.
        validation_ds: Tensorflow dataset used for validation.
    """
    plot_one(-1, train_ds, discriminator, generator)
    for epoch in range(EPOCHS):
        losses = []
        start = time.time()
        for image_batch in train_ds:
            loss = train_step(generator, discriminator, image_batch)
            losses.append(loss)

        print("Epoch " + str(epoch + 1) + ": " + str(round(time.time() - start, 3)) + " seconds")

        validate(generator, discriminator, validation_ds)
        store_loss(losses)
        plot_one(epoch, train_ds, discriminator, generator)

        # Save the model every MODEL_SAVE (from config.py) epoch
        if (epoch + 1) % MODEL_SAVE == 0:

            gen_save = "gen_model_wnet_" + str(epoch) + ".h5"
            disc_save = "disc_model_wnet_" + str(epoch) + ".h5"
            generator.save(MODEL_PATH + gen_save)
            discriminator.save(MODEL_PATH + disc_save)

            with open(MODEL_PATH + "/model.txt", "w") as f:
                f.write(gen_save + "\n")
                f.write(disc_save + "\n")
