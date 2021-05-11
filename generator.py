from model_methods import downsample, upsample
import tensorflow as tf
from config import *


def create_unet(inp):
    down_stack = [
        downsample(64, (4, 4), (2, 2), batchnorm=False),
        downsample(128, (4, 4), (2, 2)),
        downsample(256, (4, 4), (2, 2)),
        downsample(512, (4, 4), (2, 2)),
        downsample(512, (4, 4), (2, 2)),
        downsample(512, (4, 4), (2, 2)),
        downsample(512, (4, 4), (2, 2)),
    ]

    up_stack = [
        upsample(512, (4, 4), (2, 2), dropout=True),
        upsample(512, (4, 4), (2, 2), dropout=True),
        upsample(512, (4, 4), (2, 2)),
        upsample(256, (4, 4), (2, 2)),
        upsample(128, (4, 4), (2, 2)),
        upsample(64, (4, 4), (2, 2))
    ]

    x = inp
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.Conv2DTranspose(2,
                                        kernel_size=(4, 4),
                                        strides=(2, 2),
                                        padding="same",
                                        use_bias=False)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    return x


def create_generator():
    init = tf.random_normal_initializer(0., 0.02)

    inp_1 = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1])
    inp_2 = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1])

    unet_1 = create_unet(inp_1)
    unet_2 = create_unet(inp_2)

    fuse = tf.keras.layers.Concatenate()([unet_1, unet_2])

    output = tf.keras.layers.Conv2D(2,
                                    kernel_size=(4, 4),
                                    strides=(1, 1),
                                    padding="same",
                                    kernel_initializer=init,
                                    use_bias=False,
                                    activation="tanh")(fuse)

    print(output.shape)
    model = tf.keras.Model(inputs=[inp_1, inp_2], outputs=output)

    # Uncomment to visualize the model
    tf.keras.utils.plot_model(model, to_file="gen.png", show_shapes=True, dpi=64)

    return model
