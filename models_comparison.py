from main import yuv_cast
import tensorflow as tf
from tensorflow.keras import models
import random
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from config import HEIGHT, WIDTH, BATCH_SIZE
from plotting import get_channels
from main import load_data


def predict_one_img(generator, y_channel, edge):
	uv_channel = generator((y_channel, edge), training=False)
	uv_channel /= 2

	yuv_from_gen = tf.concat([y_channel, uv_channel], axis=3)
	rgb_from_gen = tf.image.yuv_to_rgb(yuv_from_gen)

	images = [rgb_from_gen[0, ...]]
	return images


def plot_row(i, original_image, wgan_100_image, img200, img300, wgan_image, fig):
	fig.add_subplot(5, 5, (i*5 + 1))
	plt.imshow(original_image[0])
	plt.axis('off')
	if i == 0:
		plt.title("Original")

	fig.add_subplot(5, 5, (i*5 + 1)+1)
	plt.imshow(wgan_100_image[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-gan 100 epochs")

	fig.add_subplot(5, 5, (i * 5 + 1) + 2)
	plt.imshow(img200[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-gan 200 epochs")

	fig.add_subplot(5, 5, (i * 5 + 1) + 3)
	plt.imshow(img300[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-gan 300 epochs")

	fig.add_subplot(5, 5, (i * 5 + 1) + 4)
	plt.imshow(wgan_image[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-gan 1000 epochs")

	return fig


def predict_one_img(generator, yuv_image, n):
	y_channel = yuv_image[..., :1]
	uv_channel = generator((y_channel, n), training=False)
	uv_channel /= 2

	yuv_from_gen = tf.concat([y_channel, uv_channel], axis=3)
	rgb_from_gen = tf.image.yuv_to_rgb(yuv_from_gen)

	return rgb_from_gen


def main():
	wgan_generator_1000 = models.load_model("models/wgan/gen_model_999.h5")
	wgan_100_generator = models.load_model("models/wgan/gen_model_99.h5")
	wgan_200_generator = models.load_model("models/wgan/gen_model_199.h5")
	wgan_300_generator = models.load_model("models/wgan/gen_model_299.h5")

	ds = load_data("data/seg_test/forest")

	fig = plt.figure()
	fig.set_figheight(15)
	fig.set_figwidth(15)

	images = []

	for img in ds.take(5):
		img = img[0, ...]
		yuv_image_tensor = tf.expand_dims(img, axis=0)
		images.append(yuv_image_tensor)

	for row in range(5):
		yuv = images[row]
		y, uv, edge = get_channels(yuv)

		rgb_wgan_100 = predict_one_img(wgan_100_generator, y, edge)
		rgb_wgan_200 = predict_one_img(wgan_200_generator, y, edge)
		rgb_wgan_300 = predict_one_img(wgan_300_generator, y, edge)
		rgb_wgan_1000 = predict_one_img(wgan_generator_1000, y, edge)

		rgb_original = tf.image.yuv_to_rgb(tf.concat([y, uv/2], axis=3))

		plot_row(row, rgb_original, rgb_wgan_100, rgb_wgan_200, rgb_wgan_300, rgb_wgan_1000, fig)

	plt.margins(1,1)
	plt.savefig("comparison.png", bbox_inches = 'tight', pad_inches = 1)
	plt.show()


if __name__ == "__main__":
    main()