# ----------------------------------------------------------------------------
# Testbench file

import sys

""" If you want to run this Neural Network at your local machine, please, 
    replace the directory described below: """

sys.path.insert(1, '/Users/elton/diagnosis/inceptionv3')

from train import *
from plot import *

# Plot Function
plot_acc_loss(train)

print('\nThe Training is Done!')