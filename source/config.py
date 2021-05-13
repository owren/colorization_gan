import tensorflow as tf
import os

# DEVELOPER VARIABLES
ENABLE_CUDA = False
DEBUG_MODE = True

# DATA INFORMATION
WIDTH = 128
HEIGHT = 128
BATCH_SIZE = 16

# TRAINING INFORMATION
EPOCHS = 1000
MODEL_SAVE = 20  # How often to save the model

# TRAINING/MODEL HYPERPARAMTERS
learning_rate = 2e-4
g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# DIRECTORY INFORMATION
experiment_name = "experiment_1"
path = os.path.abspath("../")
DATA_PATH = os.path.join(path, "data/seg_train/forest")
VALIDATION_PATH = os.path.join(path, "data/seg_validation/forest")
TEST_PATH = os.path.join(path, "data/seg_test/forest")
MODEL_PATH = os.path.join(path, "models/" + experiment_name + "/")
RESULT_PATH = os.path.join(path, "result/" + experiment_name + "/")
loss_filename = os.path.join(path, "loss/" + experiment_name + ".csv")
