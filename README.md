# Colorization using W-Net cGAN
Project for ACIT-4630: Advanced Machine Learning and Deep Learning.

Marit Øye Gjersdal, Jon-Olav Holland, Pål Anders Owren.

![WNet-cGAN](https://github.com/owren/colorization_gan/blob/main/readme_img/model.jpg?raw=true)

## Prerequisites
Linux / Windows 10

CUDA

Python 3

## Clone Repository
```
git clone https://github.com/owren/colorization_gan
cd colorization_gan/
```
## Requirements
`pip install -r requirements.txt`

## Dataset
Dataset is part of the repository in the data folder. The dataset can also be found at https://www.kaggle.com/puneet6060/intel-image-classification

## Parameters
Parameters for the dataset location, data size, enabling debugging, CUDA settings, and more, can be found in the ``source/config.py`` file.

```
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
MODEL_SAVE = 50  # How often to save the model

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
train_loss_filename = os.path.join(path, "loss/" + experiment_name + "_train.csv")
validation_loss_filename = os.path.join(path, "loss/" + experiment_name + "_validation.csv")
```
``experiment_name`` is the name of the current experiment.

``ENABLE_CUDA`` may be neccessary when running the project locally with CUDA.

``MODEL_SAVE`` indicates how often the model is saved.

## Training

```
cd source
python main.py
```
The model is saved to ``models/experiment_name/``. A text file is used to keep track of the most recently saved model.
## Testing

```
cd source
python test.py
```
The most recently saved model is used on the test dataset. The result is saved to ``result/experiment_name/``.
## Acknowledgments
Image-to-Image Translation with Conditional Adversarial Networks (https://arxiv.org/abs/1611.07004)

DSM Building Shape Refinement from Combined Remote Sensing Images based on Wnet-cGANs (https://arxiv.org/abs/1903.03519)