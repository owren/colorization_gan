import tensorflow as tf
import os
import random


def normalize(img):
    return(img - 127.5) / 127.5


filename = random.choice(os.listdir("data/seg_train/forest/sub"))
path = "data/seg_train/forest/sub/" + filename
rgb_image = tf.keras.preprocessing.image.load_img(path, target_size=(150, 150))
rgb_image = tf.keras.preprocessing.image.img_to_array(rgb_image)

rgb_image = normalize(rgb_image)
yuv_image = tf.image.rgb_to_yuv(rgb_image)
yuv_image = yuv_image.numpy()
grayscale_image = tf.image.rgb_to_grayscale(rgb_image)
grayscale_image = grayscale_image.numpy()

yuv_no_grayscale = yuv_image[:, :, 1:]

yuv_no_grayscale = tf.convert_to_tensor(yuv_no_grayscale)
grayscale_tensor = tf.convert_to_tensor(grayscale_image)

yuv_tensor = tf.concat([grayscale_tensor, yuv_no_grayscale], axis=2)
yuv_numpy = yuv_tensor.numpy()

print("x")