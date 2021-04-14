import os
import random

import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
from config import *


def discriminator_loss(real_output, gen_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    gen_loss = cross_entropy(tf.zeros_like(gen_output), gen_output)
    total_loss = real_loss + gen_loss
    return total_loss


def generator_loss(disc_gen_output, gen_output, real_output):
    gan_loss = cross_entropy(tf.ones_like(disc_gen_output), disc_gen_output)

    l1_loss = tf.reduce_mean(tf.abs(real_output - gen_output))

    total_loss = gan_loss + (100 * l1_loss)
    return total_loss


@tf.function
def train_step(generator, discriminator, images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        grayscale_batch = tf.image.rgb_to_grayscale(images)
        generated_image = generator(grayscale_batch, training=True)

        real_output = discriminator([grayscale_batch, images], training=True)
        gen_output = discriminator([grayscale_batch, generated_image], training=True)

        gen_loss = generator_loss(gen_output, generated_image, images)
        disc_loss = discriminator_loss(real_output, gen_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(generator, discriminator, dataset, epochs=10000):
    convert_random(-1, discriminator, generator)
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, image_batch)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
        convert_random(epoch, discriminator, generator)


def convert_random(epoch, discriminator, generator):
    filename = random.choice(os.listdir("data/seg_train/forest/sub"))
    path = "data/seg_train/forest/sub/" + filename

    rgb_image = tf.keras.preprocessing.image.load_img(path, target_size=(HEIGHT, WIDTH))
    rgb_image_tensor = tf.keras.preprocessing.image.img_to_array(rgb_image)
    rgb_image_tensor = tf.expand_dims(rgb_image_tensor, axis=0)
    grayscale_image = tf.image.rgb_to_grayscale(rgb_image_tensor)

    gen_image = generator(grayscale_image, training=False)
    images = [rgb_image, gen_image[0, ...], (gen_image[0] * 127.5) + 127.5]

    fig = plt.figure()
    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        plt.axis("off")
        plt.imshow(images[i])

    plt.text(-140, -45, "Epoch: " + str(epoch), fontsize=18)
    plt.show()
