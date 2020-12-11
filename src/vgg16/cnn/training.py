# ----------------------------------------------------------------------------
# Training Module

from model import *

print("[INFO] TRAINING HEAD...")

print("\n")

# Partition the data into training and testing splits using 80% of the data
# for training and the remaining 20% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size = 0.20, stratify = lables, random_state = 42)

# Initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15,
	                          fill_mode="nearest")

result = model.fit_generator(trainAug.flow(trainX, trainY, batch_size=BATCH),
	                         steps_per_epoch=len(trainX) // BATCH,
	                         validation_data=(testX, testY),
	                         validation_steps=len(testX) // BATCH,
	                         epochs=EPOCHS)

# ----------------------------------------------------------------------------
# Exporting Saved Models

model_version = "0001"
model_name    = "vgg16"
model_path    = os.path.join(model_name, model_version)

# HDF5 FORMAT
model.save('vgg16.h5')

# Model's computation graph and its weigths
tf.saved_model.save(model, model_path)

