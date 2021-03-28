# ----------------------------------------------------------------------------
# Parameters for Training and Validation

import sys

""" If you want to run this Neural Network at your local machine, please, 
    replace the directory described below: """

sys.path.insert(1, '/Users/elton/diagnosis/inceptionv3')

from lib import *

BASE_DIR  = '/Users/elton/diagnosis/data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')

# Parameters for Training and Validation

IMAGE_SIZE    = (150, 150)
BATCH         = 32
EPOCHS        = 60
LEARNING_RATE = 1e-5
DECAY         = 1e-6