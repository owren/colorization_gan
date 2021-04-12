import os
import random

import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np


def discriminator_loss(cross_entropy, real_output, gen_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    gen_loss = cross_entropy(tf.zeros_like(gen_output), gen_output)
    total_loss = real_loss + gen_loss
    return total_loss


def generator_loss(cross_entropy, gen_output):
    return cross_entropy(tf.ones_like(gen_output), gen_output)


@tf.function
def train_step(generator, discriminator, images, cross_entropy, g_optimizer, d_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(tf.image.rgb_to_grayscale(images), training=True)

        real_output = discriminator(images, training=True)
        gen_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(cross_entropy, gen_output)
        disc_loss = discriminator_loss(cross_entropy, real_output, gen_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(generator, discriminator, dataset, cross_entropy, g_optimizer, d_optimizer, epochs=10000):
    convert_random(-1, discriminator, generator)
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, image_batch, cross_entropy, g_optimizer, d_optimizer)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
        convert_random(epoch, discriminator, generator)


def convert_random(epoch, discriminator, generator):
    filename = random.choice(os.listdir("data/seg_train/forest/sub"))
    path = "data/seg_train/forest/sub/" + filename
    fig = plt.figure()

    rgb_image = tf.keras.preprocessing.image.load_img(path, target_size=(150, 150))
    rgb_image_tensor = tf.keras.preprocessing.image.img_to_array(rgb_image)
    real_predict = discriminator.predict(tf.expand_dims(rgb_image_tensor, axis=0))

    fig.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.text(50, -10, str(round(real_predict[0, 0] * 100, 1)) + "%", fontsize=16)
    plt.text(125, -45, "Epoch: " + str(epoch), fontsize=18)
    plt.imshow(rgb_image_tensor/255)

    grayscale_image = tf.image.rgb_to_grayscale(rgb_image_tensor)
    grayscale_image = tf.expand_dims(grayscale_image, axis=0)
    gen_image = generator(grayscale_image, training=False)
    gen_predict = discriminator.predict(gen_image)

    fig.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.text(50, -10, str(round(gen_predict[0, 0] * 100, 1)) + "%", fontsize=16)
    plt.imshow(np.uint8(gen_image[0, :, :, :] * 127.5 + 127.5))

    plt.show()
