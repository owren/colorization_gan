from model_methods import downsample, upsample
import tensorflow as tf
from config import *


def create_generator():
    inp = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1])

    stack = [
        downsample(64, (3, 3), (1, 1), batchnorm=False),
        downsample(128, (3, 3), (1, 1)),
        downsample(256, (3, 3), (1, 1)),
        downsample(512, (3, 3), (1, 1)),
        downsample(512, (3, 3), (1, 1)),
        downsample(512, (3, 3), (1, 1)),
        downsample(512, (3, 3), (1, 1)),
        downsample(512, (3, 3), (1, 1)),
        downsample(1024, (3, 3), (1, 1))
    ]

    output = tf.keras.layers.Conv2DTranspose(2,
                                             kernel_size=(4, 4),
                                             strides=(1, 1),
                                             padding="same",
                                             use_bias=False,
                                             activation="tanh")

    x = inp
    skips = []

    for i in range(len(stack) - 2):
        next_layer = stack[i](x)

        if i % 2 == 0:
            conc = tf.keras.layers.Concatenate()([x, next_layer])
            x = conc
        else:
            x = next_layer

    x = output(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    # Uncomment to visualize the model
    # tf.keras.utils.plot_model(model, to_file="gen.png", show_shapes=True, dpi=64)
    # model.summary()

    return model
