
# ----------------------------------------------------------------------------
# Create Data Generators to preprocess and prepare training and validationimport sys

import sys

sys.path.insert(1, '/Users/elton/diagnosis/src/lib')
sys.path.insert(1, '/Users/elton/diagnosis/src/parameters')

from lib import *
from tf_lib import *
from parameters import *

# Grab the list of images in our datasetdirectory

print("\n")
print("[INFO] LOADING IMAGES...")
print("\n")

imagePaths = list(paths.list_images(DATASET_PATH))
data   = []
labels = []

# Loop over the image paths
for imagePath in imagePaths:
	# Extract the class label from the filename
	label = imagePath.split(os.path.sep)[-3]

	# Load the image, swap color channels, and resize it to be a fixed at
	# 224x224 pixels while ignoring aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# Update the data and labels lists
	data.append(image)
	labels.append(label)

# Convert the data and lables to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]

data   = np.array(data) / 225.0
lables = np.array(labels)

# Perform One-Hot encoding 

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)



