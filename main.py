# ----------------------------------------------------------------------------
# File dedicated for functions used in the VGG-CAM structure

import sys

""" If you want to run this Neural Network at your local machine, please, replace the directory described below: """

sys.path.insert(1, '/Users/elton/diagnosis/parameters')
sys.path.insert(1, '/Users/elton/diagnosis/lib')

from parameters import *
from def_cam import *

weights_path = '/Users/elton/diagnosis/cam/vgg16_weights_th_dim_ordering_th_kernels.h5'

test = train_VGGCAM(weights_path, NB_CLASSES, NUM_INPUT_CHANNELS)

print('I am HERE!')