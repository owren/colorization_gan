import os
import random

import tensorflow as tf
import time
import matplotlib.pyplot as plt


def discriminator_loss(cross_entropy, real_output, gan_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(gan_output), gan_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(cross_entropy, gan_output):
    return cross_entropy(tf.ones_like(gan_output), gan_output)


@tf.function
def train_step(generator, discriminator, images, cross_entropy, g_optimizer, d_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(tf.image.rgb_to_grayscale(images), training=True)

        real_output = discriminator(images, training=True)
        gan_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(cross_entropy, gan_output)
        disc_loss = discriminator_loss(cross_entropy, real_output, gan_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(generator, discriminator, dataset, cross_entropy, g_optimizer, d_optimizer, epochs=100):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, image_batch, cross_entropy, g_optimizer, d_optimizer)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
        convert_random(generator)


def convert_random(generator):
    filename = random.choice(os.listdir("data/seg_train/forest/sub"))
    path = "data/seg_train/forest/sub/" + filename

    rgb_image = tf.keras.preprocessing.image.load_img(path, target_size=(150, 150))
    rgb_image_tensor = tf.keras.preprocessing.image.img_to_array(rgb_image)
    plt.imshow(rgb_image_tensor/255)
    plt.axis("off")
    plt.show()

    grayscale_image = tf.image.rgb_to_grayscale(rgb_image_tensor)
    grayscale_image = tf.expand_dims(grayscale_image, axis=0)
    gan_image = generator(grayscale_image, training=False)

    plt.imshow(gan_image[0, :, :, 0] * 127.5 + 127.5)
    plt.axis("off")
    plt.show()