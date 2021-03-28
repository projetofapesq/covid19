# ----------------------------------------------------------------------------
# File dedicated for functions used in the VGG-CAM structure

import sys

""" If you want to run this Neural Network at your local machine, please, you 
have to replace the directory described below: """

sys.path.insert(1, '/Users/elton/diagnosis/lib')

from lib import *
from tf_lib import *
from cam_lib import *

# Defining functions

def VGGCAM(nb_classes, num_input_channels):
	"""
	Build Convolution Neural Network in which returns model (Keras NN) the Neural Net model
	"""

	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape = (3, 224, 224)))
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation = 'relu'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, activation = 'relu'))
	"""
	Add another convolutional layer with ReLU  + GAP
	"""
	model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', padding='same'))
	model.add(AveragePooling2D(pool_size = (14, 14), strides = None, padding = 'same', data_format = None))
	model.add(Flatten())

	"""
	Add W Layer
	"""
	model.add(Dense(nb_classes, activation = 'softmax'))

	model._name = "VGGCAM"

	return model

def train_VGGCAM(VGG_weights_path, nb_classes, num_input_channels):
	"""
	Train VGG-CAM Model
	Args: VGG_weight_path (str) path to Keras VGG16 weights
	      nb_classes (int) number of classes
	      num_input_channels (int) number of convolutional filters to add in before the GAP layers
	"""
	
	# Load Model
	model = VGGCAM(nb_classes, num_input_channels)
	modelVGG = VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))


















