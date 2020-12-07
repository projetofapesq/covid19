# ----------------------------------------------------------------------------
# Import Files

from preprocess import *

# Set Optimizer

opt = Adam(lr = LEARNING_RATE, decay = DECAY)

# Simple Convolutional Neural Network (CNN) based on Xception

conv_base = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

conv_base.trainable  = True

# Set Model

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(4, activation = 'softmax'))

# Compile Model

model.compile(loss='categorical_crossentropy', 
	          optimizer = opt,
	          metrics=['accuracy'])

#Summary of Xception Model

print(conv_base.summary())               