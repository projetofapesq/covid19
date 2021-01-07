# ----------------------------------------------------------------------------
# Parameters for Training and Validation

DATASET_PATH  = '/Users/elton/diagnosis/data/train'
TEST_DIR      = '/Users/elton/diagnosis/data/test'

IMAGE_SIZE    = (150, 150)
BATCH         = 10
EPOCHS        = 80
DECAY         = 1e-6
LEARNING_RATE = 0.0001

# ----------------------------------------------------------------------------
# Parameters for VGG-CAM Network

NB_CLASSES         = 1000
NUM_INPUT_CHANNELS = 1024
