import tensorflow as tf
import time

# DEVELOPER VARIABLES
ENABLE_CUDA = True
DEBUG_MODE = True

# DATA INFORMATION
WIDTH = 128
HEIGHT = 128
BATCH_SIZE = 16

# TRAINING INFORMATION
learning_rate = 2e-4
EPOCHS = 1000
g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# DIRECTORY INFORMATION
DATA_PATH = "data/seg_train/forest/sub"
timestr = time.strftime("%Y%m%d-%H%M%S")
loss_filename = "loss/losses_" + timestr + ".csv"
