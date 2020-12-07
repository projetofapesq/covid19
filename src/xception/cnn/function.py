# ----------------------------------------------------------------------------
# Evaluation: plot the accuracy and loss graph

import sys

sys.path.insert(1, '/Users/elton/diagnosis/src/lib')

from lib import *

def plot_acc_loss(result, epochs):
	plt.style.use("seaborn-white")
	plt.figure()
	plt.plot(np.arange(0, epochs), result.history["loss"], label = 'Training Loss')
	plt.plot(np.arange(0, epochs), result.history["val_loss"], label = 'Validation Loss')
	plt.plot(np.arange(0, epochs), result.history["accuracy"], label = 'Training Accuracy')
	plt.plot(np.arange(0, epochs), result.history["val_accuracy"], label = 'Validation Accuracy')
	plt.title("Training Loss and Accuracy Dataset")
	plt.xlabel("Epoch")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="center right")
	plt.savefig("plot.png")
	plt.show()