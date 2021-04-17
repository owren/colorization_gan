from config import *


def downsample(filters, kernel_size, strides, batchnorm=True):
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2D(filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding="same",
                                     use_bias=False))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    layer.add(tf.keras.layers.LeakyReLU())

    return layer


def upsample(filters, kernel_size, strides, dropout=False):
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2DTranspose(filters,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding="same",
                                              use_bias=False))

    layer.add(tf.keras.layers.BatchNormalization())

    if dropout:
        layer.add(tf.keras.layers.Dropout(0.5))

    layer.add(tf.keras.layers.LeakyReLU())

    return layer


# TODO: Possible to move concate of grayscale and YUV variables to the model?
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

    tf.keras.utils.plot_model(model, to_file="gen.png", show_shapes=True, dpi=64)

    return model


def create_discriminator():
    inp = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1], name="input_image")
    tar = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 2], name="target_image")

    input_layer = tf.keras.layers.concatenate([inp, tar])

    c1 = downsample(32, (4, 4), (2, 2))(input_layer)
    c2 = downsample(64, (4, 4), (2, 2))(c1)
    c3 = downsample(128, (4, 4), (2, 2))(c2)

    zero_pad_1 = tf.keras.layers.ZeroPadding2D()(c3)
    c4 = downsample(256, (4, 4), (1, 1))(zero_pad_1)

    zero_pad_2 = tf.keras.layers.ZeroPadding2D()(c4)

    # TODO: Sigmoid activation function?
    output = tf.keras.layers.Conv2D(1,
                                    kernel_size=(4, 4),
                                    strides=(1, 1),
                                    padding="same",
                                    activation="sigmoid",
                                    use_bias=False)(zero_pad_2)

    model = tf.keras.Model(inputs=[inp, tar], outputs=output)

    # tf.keras.utils.plot_model(model, to_file="disc.png", show_shapes=True, dpi=64)

    return model