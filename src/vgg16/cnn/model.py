# ----------------------------------------------------------------------------
# Import Files

from preprocess import *

# Set Optimizer

opt = Adam(lr = LEARNING_RATE, decay = DECAY)

# Simple Convolutional Neural Network (CNN) based on VGG-16

baseModel = VGG16(weights = 'imagenet', include_top = False,
	              input_shape = (224, 224, 3))

# Construct the head of the model that will be placed on top of the base model

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4,4))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(64, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = "softmax")(headModel)

# Compile Model

print("\n")

print("[INFO] COMPILING MODEL...")

print("\n")

# Place the head model on top of the base model
# This will become the actual model that will be trained

model = Model(inputs = baseModel.input, outputs = headModel)

model.compile(loss = 'binary_crossentropy',
	          optimizer = opt,
	          metrics = ['accuracy'])

# Summary of VGG-16

print(baseModel.summary()) 

print("\n")

# Loop over all layers in the base model and freeze them. So they will not be
# updated during the first training process

for layer in baseModel.layers:
	layer.trainable = False

