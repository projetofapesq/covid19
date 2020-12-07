# ----------------------------------------------------------------------------
# Create Data Generators to preprocess and prepare training and validation

import sys

sys.path.insert(1, '/Users/elton/diagnosis/src/lib')

from lib import *
from tf_lib import *
from parameters import *

# Preprocessor

train_datagen = ImageDataGenerator(rescale = 1./255,
	                               rotation_range = 50,
	                               featurewise_center = True,
	                               featurewise_std_normalization = True,
	                               width_shift_range = 0.2,
	                               height_shift_range = 0.2,
	                               shear_range = 0.25,
	                               zoom_range = 0.1,
	                               zca_whitening = True,
	                               channel_shift_range = 20,
	                               horizontal_flip = True,
	                               vertical_flip = True,
	                               validation_split = 0.2,
	                               fill_mode = 'constant'
	                               )

# For Categorical Muticlass

train_batches = train_datagen.flow_from_directory(DATASET_PATH,
	                                              target_size = IMAGE_SIZE,
	                                              shuffle = True,
	                                              batch_size = BATCH,
	                                              subset = "training",
	                                              seed = 42,
	                                              class_mode = "categorical")

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,
	                                              target_size = IMAGE_SIZE,
	                                              shuffle = True,
	                                              batch_size = BATCH,
	                                              subset = "validation",
	                                              seed = 42,
	                                              class_mode = "categorical")