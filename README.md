# Colorization with W-Net
Project for ACIT-4630: Advanced Machine Learning and Deep Learning.

## Prerequisites
Linux

CUDA

Python 3

## Requirements
`pip install -r requirements.txt`

## Dataset
Dataset is part of the repository in the data folder. The dataset can also be found at https://www.kaggle.com/puneet6060/intel-image-classification

## Parameters
Parameters for the dataset location, data size, enabling debugging, CUDA settings, and more, can be found in the ``config.py`` file.

```
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
d_optimizer = tf.keras .optimizers.Adam(learning_rate, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# DIRECTORY INFORMATION
DATA_PATH = "data/seg_train/forest/sub"
timestr = time.strftime("%Y%m%d-%H%M%S")
loss_filename = "loss/losses_" + timestr + ".csv"
```
