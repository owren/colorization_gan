from model_methods import downsample, upsample
import tensorflow as tf
from config import *


def create_generator():
    inp = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1])

    down_stack = [
        downsample(64, (4, 4), (2, 2), batchnorm=False),
        downsample(128, (4, 4), (2, 2)),
        downsample(256, (4, 4), (2, 2)),
        downsample(256, (4, 4), (2, 2))
    ]

    up_stack = [
        upsample(256, (4, 4), (2, 2), dropout=True),
        upsample(128, (4, 4), (2, 2), dropout=True),
        upsample(64, (4, 4), (2, 2))
    ]

    output = tf.keras.layers.Conv2DTranspose(2,
                                             kernel_size=(4, 4),
                                             strides=(2, 2),
                                             padding="same",
                                             use_bias=False,
                                             activation="tanh")

    x = inp
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = output(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    # Uncomment to visualize the model
    tf.keras.utils.plot_model(model, to_file="gen.png", show_shapes=True, dpi=64)

    return model
