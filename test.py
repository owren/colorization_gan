import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt

# Irrelevant test file

filename = random.choice(os.listdir("data/seg_train/forest/sub"))
path = "data/seg_train/forest/sub/" + filename
rgb_image = tf.keras.preprocessing.image.load_img(path, target_size=(150, 150))
rgb_image = tf.keras.preprocessing.image.img_to_array(rgb_image)

rgb_image /= 255.

yuv_image = tf.image.rgb_to_yuv(rgb_image).numpy()
y = yuv_image[..., 0]
u = yuv_image[..., 1]
v = yuv_image[..., 2]


images = [rgb_image]

fig = plt.figure()
for i in range(len(images)):
    fig.add_subplot(1, len(images), i + 1)
    plt.axis("off")
    plt.imshow(images[i])


plt.show()

'''
sigmoid_ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
categorical_ce = tf.keras.losses.CategoricalCrossentropy()
true = tf.constant([1., 1., 1.])
predict = tf.constant([1., 1., 1.])
loss = sigmoid_ce(true, predict)
print(loss)

'''