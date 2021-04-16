import tensorflow as tf


WIDTH = 128
HEIGHT = 128
BATCH_SIZE = 32
EPOCHS = 1000

learning_rate = 2e-4
g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
