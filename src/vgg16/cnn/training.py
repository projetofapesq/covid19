# ----------------------------------------------------------------------------
# Training Module

from model import *

print("[INFO] TRAINING HEAD...")

print("\n")


result = model.fit_generator(
	                         )

# ----------------------------------------------------------------------------
# Exporting Saved Models

model_version = "0001"
model_name    = "vgg16"
model_path    = os.path.join(model_name, model_version)

# HDF5 FORMAT
model.save('vgg16.h5')

# Model's computation graph and its weigths
tf.saved_model.save(model, model_path)

