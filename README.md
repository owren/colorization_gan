# Colorization with W-Net
Project for ACIT-4630: Advanced Machine Learning and Deep Learning.

Marit Øye Gjersdal, Jon-Olav Holland, Pål Anders Owren.

## Prerequisites
Linux / Windows 10

CUDAs

Python 3

## Requirements
`pip install -r requirements.txt`

## Dataset
Dataset is part of the repository in the data folder. The dataset can also be found at https://www.kaggle.com/puneet6060/intel-image-classification

## Parameters
Parameters for the dataset location, data size, enabling debugging, CUDA settings, and more, can be found in the ``source/config.py`` file.

```
import tensorflow as tf
import time
import os

# DEVELOPER VARIABLES
ENABLE_CUDA = False
DEBUG_MODE = True

# DATA INFORMATION
WIDTH = 128
HEIGHT = 128
BATCH_SIZE = 16

# TRAINING INFORMATION
learning_rate = 2e-4
EPOCHS = 1000
MODEL_SAVE = 50
g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# DIRECTORY INFORMATION
path = os.path.abspath("../")
DATA_PATH = os.path.join(path, "data/seg_train/forest/sub")
timestr = time.strftime("%Y%m%d-%H%M%S")
loss_filename = os.path.join(path, "loss/losses_" + timestr + ".csv")
```
## Training

```
python main.py
```
The model is saved to ``models/experiment_name/``
## Testing

```
python test.py
```
The result is saved to ``result/experiment_name/``
## Acknowledgments
Image-to-Image Translation with Conditional Adversarial Networks (https://arxiv.org/abs/1611.07004)
W-Net: A Deep Model for Fully Unsupervised Image Segmentation (https://arxiv.org/abs/1711.08506)