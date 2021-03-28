# ----------------------------------------------------------------------------
# Evaluation: plot the accuracy and loss graph

import sys

""" If you want to run this Neural Network at your local machine, please, 
    replace the directory described below: """

sys.path.insert(1, '/Users/elton/Documents/FAPESQ/InceptionV3')

from lib import *

def plot_acc_loss(result):
	acc      = result.history['acc']
	val_acc  = result.history['val_acc']
	loss     = result.history['loss']
	val_loss = result.history['val_loss']

	epochs   = range(len(acc))
	plt.figure()
	plt.plot(epochs, acc, c='b', label = 'Training Accuracy')
	plt.plot(epochs, val_acc, c='r', label = 'Validation Accuracy')
	plt.plot(epochs, loss, c='g', label = 'Training Loss')
	plt.plot(epochs, val_loss, c='m', label = 'Validation Loss')
	plt.title("Training and Validation Accuracy/Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy/Loss")
	plt.legend(loc = "center right")
	plt.savefig("plot_v6.png")
	plt.show()		