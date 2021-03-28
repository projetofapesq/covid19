# ----------------------------------------------------------------------------
# Preprocess Image File Function

import sys

""" If you want to run this Neural Network at your local machine, please, 
    replace the directory described below: """

sys.path.insert(1, '/Users/elton/Documents/FAPESQ/InceptionV3')

from parameters import *

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), #Shape of our images
	                            include_top = False,         # Leave out the last fully connected layer
	                            weights     = 'imagenet')

'''
Make all the layers non trainable. We can retrain some of the lower layers to increase performance.
'''

for layer in pre_trained_model.layers:
	layer.trainable = False

'''
Customized early stopping
'''

custom_early_stopping =  EarlyStopping(monitor   = 'val_acc', # Performance measure to terminate the training
									   patience  = 10,        # Number of epochs with no improvement
									   min_delta = 0.001,     
									   mode      = 'max')



# Flatten the output layer to 1 dimension
x = layers.Flatten()(pre_trained_model.output)

# Add a fully connected layer with 1.024 hidden units and ReLU activation
x = layers.Dense(1024, activation = 'relu')(x)

# Add a dropout rate 
x = layers.Dropout(0.5)(x) # or 0.2

# Add a final sigmoid layer for classification
x = layers.Dense(5, activation = 'softmax')(x)

model = Model(pre_trained_model.input, x)

# Set Optimizer

opt = Adam(lr = LEARNING_RATE, decay = DECAY)

# ----------------------------------------------------------------------------

print("\n[INFO] COMPILING MODEL...\n")

model.compile(optimizer = opt,
			  loss      = 'categorical_crossentropy', # modificar
			  metrics   = ['acc'])

#Summary of InceptionV3 Model

print(pre_trained_model.summary())  