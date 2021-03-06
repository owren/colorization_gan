import tensorflow as tf
from config import HEIGHT, WIDTH
from model_methods import downsample


def create_discriminator():
    """Creates a discriminiator similiar to PatchGAN.

    Returns:
        A Keras model.
    """
    inp = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 1], name="input_image")
    tar = tf.keras.layers.Input(shape=[HEIGHT, WIDTH, 2], name="target_image")

    input_layer = tf.keras.layers.concatenate([inp, tar])

    c1 = downsample(64, (4, 4), (2, 2))(input_layer)
    c2 = downsample(128, (4, 4), (2, 2))(c1)
    c3 = downsample(256, (4, 4), (2, 2))(c2)

    zero_pad_1 = tf.keras.layers.ZeroPadding2D()(c3)
    c4 = downsample(512, (4, 4), (1, 1))(zero_pad_1)

    zero_pad_2 = tf.keras.layers.ZeroPadding2D()(c4)

    output = tf.keras.layers.Conv2D(1,
                                    kernel_size=(4, 4),
                                    strides=(1, 1),
                                    padding="same",
                                    activation="sigmoid",
                                    use_bias=False)(zero_pad_2)

    model = tf.keras.Model(inputs=[inp, tar], outputs=output)

    # Uncomment to visualize the model
    # tf.keras.utils.plot_model(model, to_file="disc.png", show_shapes=True, dpi=64)

    return model
