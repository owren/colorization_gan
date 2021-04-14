import tensorflow as tf


WIDTH = 128
HEIGHT = 128
BATCH_SIZE = 32

learning_rate = 2e-4
g_optimizer = tf.keras.optimizers.Adam(learning_rate)
d_optimizer = tf.keras.optimizers.Adam(learning_rate)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
