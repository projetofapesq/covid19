# ----------------------------------------------------------------------------
# Training Module

from model import *

# FIT MODEL and TRAINING

print(len(train_batches))
print(len(valid_batches))

print("\n")

print("[INFO] COMPILING MODEL...")

print("\n")

STEP_SIZE_TRAIN = train_batches.n//train_batches.batch_size
STEP_SIZE_VALID = valid_batches.n//valid_batches.batch_size

print("[INFO] TRAINING HEAD...")

print("\n")

result = model.fit_generator(train_batches,
	                         steps_per_epoch = STEP_SIZE_TRAIN,
	                         validation_data = valid_batches,
	                         validation_steps = STEP_SIZE_VALID,
	                         epochs = EPOCHS)

# ----------------------------------------------------------------------------
# Exporting Saved Models
model_version = "0001"
model_name    = "xception"
model_path    = os.path.join(model_name, model_version)

# HDF5 FORMAT
model.save('xception.h5')

# Model's computation graph and its weigths
tf.saved_model.save(model, model_path)