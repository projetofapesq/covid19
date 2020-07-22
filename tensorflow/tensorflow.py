import tensorflow as tf
import keras
import sklearn

# Import the necessary packages from TensorFlow, Keras, scikit-learn and imutils

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.optimezers import Adam
from keras.utils import to_categorical
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imutils import paths