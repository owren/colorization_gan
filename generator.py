from model_methods import downsample, upsample
import tensorflow as tf
from config import *


def create_generator():
    inp = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1])
    init = tf.random_normal_initializer(0., 0.02)

    t0 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), padding="same", kernel_initializer=init, use_bias=False)(inp)
    t0 = tf.keras.layers.LeakyReLU()(t0)
    t0 = tf.keras.layers.BatchNormalization()(t0)

    t1 = tf.keras.layers.Concatenate()([inp, t0])
    t1 = tf.keras.layers.Conv2D(128, kernel_size=(5, 5), padding="same", kernel_initializer=init, use_bias=False)(t1)
    t1 = tf.keras.layers.LeakyReLU()(t1)
    t1 = tf.keras.layers.Dropout(0.5)(t1)
    t1 = tf.keras.layers.BatchNormalization()(t1)

    t2 = tf.keras.layers.Concatenate()([inp, t1])
    t2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), padding="same", kernel_initializer=init, use_bias=False)(t2)
    t2 = tf.keras.layers.LeakyReLU()(t2)
    t2 = tf.keras.layers.Dropout(0.5)(t2)
    t2 = tf.keras.layers.BatchNormalization()(t2)

    t3 = tf.keras.layers.Concatenate()([inp, t2])
    t3 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), padding="same", kernel_initializer=init, use_bias=False)(t3)
    t3 = tf.keras.layers.LeakyReLU()(t3)
    t3 = tf.keras.layers.BatchNormalization()(t3)

    t4 = tf.keras.layers.Concatenate()([inp, t3])
    output = tf.keras.layers.Conv2D(2,
                                    kernel_size=(5, 5),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_initializer=init,
                                    use_bias=False,
                                    activation="tanh")(t4)

    model = tf.keras.Model(inputs=inp, outputs=output)

    # Uncomment to visualize the model
    tf.keras.utils.plot_model(model, to_file="gen.png", show_shapes=True, dpi=64)
    model.summary()

    return model

