# ----------------------------------------------------------------------------
# Train Module Functions

import sys

""" If you want to run this Neural Network at your local machine, please, 
    replace the directory described below: """

sys.path.insert(1, '/Users/elton/Documents/FAPESQ/InceptionV3')

from preprocess import *

# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(rescale            = 1./255.,
								   rotation_range     = 40,
								   width_shift_range  = 0.2,
								   height_shift_range = 0.2,
								   shear_range        = 0.2,
								   zoom_range         = 0.2,
								   horizontal_flip    = True)

# Note that the validation data should not be augmented

test_datagen = ImageDataGenerator(rescale = 1./255.)

# Flow training images in batches using train_datagen generator

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
													batch_size = BATCH,
													class_mode = 'categorical', 
													target_size = IMAGE_SIZE)

validation_generator = test_datagen.flow_from_directory(TEST_DIR,
														batch_size = BATCH,
														class_mode = 'categorical',
														target_size = IMAGE_SIZE)

print("\n[INFO] TRAINING AHEAD...\n")

# FIT MODEL and TRAINING

print('Length - Train Generator:', len(train_generator))
print('Length - Validation Generator:', len(validation_generator))

print("\n")

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

train     = model.fit(train_generator,
					  validation_data  = validation_generator,
					  steps_per_epoch  = STEP_SIZE_TRAIN,
					  epochs           = EPOCHS,
					  validation_steps = STEP_SIZE_VALID,
					  verbose = 2,
					  callbacks = [custom_early_stopping])

# ----------------------------------------------------------------------------
# Exporting Saved Models

model_version = "0006"
model_name    = "InceptionV3"
model_path    = os.path.join(model_name, model_version)

# HDF5 FORMAT
model.save('model_v6.h5')

# Model's computation graph and its weigths
tf.saved_model.save(model, model_path)