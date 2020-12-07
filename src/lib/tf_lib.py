# ----------------------------------------------------------------------------
# Import TensorFlow Libraries and Keras Packages

import tensorflow

# TensorFlow Packages 

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.applications import Xception   # TensorFlow only
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

# Keras on top of TensorFlow 2.0  

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# ----------------------------------------------------------------------------

from numpy.random import seed
seed(8)

tensorflow.random.set_seed(7)

import tensorflow as tf