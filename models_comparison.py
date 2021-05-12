from main import yuv_cast
import tensorflow as tf
from tensorflow.keras import models
import random
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from config import HEIGHT, WIDTH, BATCH_SIZE
from utility import get_channels
from main import load_data


def predict_one_img(generator, y_channel, edge):
	uv_channel = generator((y_channel, edge), training=False)
	uv_channel /= 2

	yuv_from_gen = tf.concat([y_channel, uv_channel], axis=3)
	rgb_from_gen = tf.image.yuv_to_rgb(yuv_from_gen)

	images = [rgb_from_gen[0, ...]]
	return images


def plot_row(i, original_image, wnet_100_image, img200, img300, wnet_image, fig):
	fig.add_subplot(5, 5, (i*5 + 1))
	plt.imshow(original_image[0])
	plt.axis('off')
	if i == 0:
		plt.title("Original")

	fig.add_subplot(5, 5, (i*5 + 1)+1)
	plt.imshow(wnet_100_image[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-net 100 epochs")

	fig.add_subplot(5, 5, (i * 5 + 1) + 2)
	plt.imshow(img200[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-net 200 epochs")

	fig.add_subplot(5, 5, (i * 5 + 1) + 3)
	plt.imshow(img300[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-net 300 epochs")

	fig.add_subplot(5, 5, (i * 5 + 1) + 4)
	plt.imshow(wnet_image[0])
	plt.axis('off')
	if i == 0:
		plt.title("W-net 1000 epochs")

	return fig


def predict_one_img(generator, yuv_image, n):
	y_channel = yuv_image[..., :1]
	uv_channel = generator((y_channel, n), training=False)
	uv_channel /= 2

	yuv_from_gen = tf.concat([y_channel, uv_channel], axis=3)
	rgb_from_gen = tf.image.yuv_to_rgb(yuv_from_gen)

	return rgb_from_gen


def main():
	wnet_generator_1000 = models.load_model("models/wnet/gen_model_999.h5")
	wnet_100_generator = models.load_model("models/wnet/gen_model_99.h5")
	wnet_200_generator = models.load_model("models/wnet/gen_model_199.h5")
	wnet_300_generator = models.load_model("models/wnet/gen_model_299.h5")

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

		rgb_wnet_100 = predict_one_img(wnet_100_generator, y, edge)
		rgb_wnet_200 = predict_one_img(wnet_200_generator, y, edge)
		rgb_wnet_300 = predict_one_img(wnet_300_generator, y, edge)
		rgb_wnet_1000 = predict_one_img(wnet_generator_1000, y, edge)

		rgb_original = tf.image.yuv_to_rgb(tf.concat([y, uv/2], axis=3))

		plot_row(row, rgb_original, rgb_wnet_100, rgb_wnet_200, rgb_wnet_300, rgb_wnet_1000, fig)

	plt.margins(1,1)
	plt.savefig("comparison.png", bbox_inches = 'tight', pad_inches = 1)
	plt.show()


if __name__ == "__main__":
    main()