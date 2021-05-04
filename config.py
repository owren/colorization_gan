import tensorflow as tf
import os
import time

WIDTH = 128
HEIGHT = 128
BATCH_SIZE = 32
EPOCHS = 1000

learning_rate = 2e-4
g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


timestr = time.strftime("%Y%m%d-%H%M%S")
loss_filename = "loss/losses_" + timestr + ".csv"

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

