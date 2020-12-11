# ----------------------------------------------------------------------------
# Import TensorFlow Libraries and Keras Packages

import tensorflow

# TensorFlow Packages 

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, AveragePooling2D
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.applications import Xception, VGG16  
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

# Sklearn on top of TensorFlow 2.0 necessary packages

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Keras on top of TensorFlow 2.0 necessary packages

from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# ----------------------------------------------------------------------------

from numpy.random import seed
seed(8)

tensorflow.random.set_seed(7)

import tensorflow as tf