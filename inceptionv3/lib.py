# ----------------------------------------------------------------------------
# Import Python Libraries, Keras and TensorFlow Packages

import os
import tensorflow as tf
import numpy as np               # Linear Algebra
import pandas as pd              # Data processing and CSV file I/O
import cv2                       # OpenCV library
import argparse
import matplotlib.pyplot as plt  # Graphical Processing

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping

from skimage import transform

# Import the Inception V3 Model

from tensorflow.keras.applications.inception_v3 import InceptionV3

print('\nHello! We are using TensorFlow version:', tf.__version__ + '\n')