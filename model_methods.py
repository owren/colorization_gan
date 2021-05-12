import tensorflow as tf


def downsample(filters, kernel_size, strides, batchnorm=True):
    """A downsample block for both the generator and discriminator.

    Args:
        filters: An integer count of the filters in the Convolutional Tranpose layer.
        kernel_size: An integer pair of the kernel size.
        strides: An integerp pair of the stride size.
        batchnorm: A boolean indicating if Batch Normalization should be enabled.

    Returns:
        A sequential model which consists of a Convolutional layer, a Batch Normalization layer
        (if enabled), and a Leakyrelu layer.
    """
    init = tf.random_normal_initializer(0., 0.02)
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2D(filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding="same",
                                     kernel_initializer=init,
                                     use_bias=False))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    layer.add(tf.keras.layers.LeakyReLU())

    return layer


def upsample(filters, kernel_size, strides, dropout=False):
    """An upsample block for the generator.

    Args:
        filters: An integer count of the filters in the Convolutional Tranpose layer.
        kernel_size: An integer pair of the kernel size.
        strides: An integerp pair of the stride size.
        dropout: A boolean indicating if dropout should be enabled.

    Returns:
        A sequential model which consists of a Convolutional Tranpose layer, a Batch Normalization layer,
        a Dropout layer (if enabled), and a Leakyrelu layer.
    """
    init = tf.random_normal_initializer(0., 0.02)
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(filters,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding="same",
                                              kernel_initializer=init,
                                              use_bias=False))

    block.add(tf.keras.layers.BatchNormalization())

    if dropout:
        block.add(tf.keras.layers.Dropout(0.5))

    block.add(tf.keras.layers.LeakyReLU())

    return block
