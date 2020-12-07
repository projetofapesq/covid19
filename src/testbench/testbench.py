# ----------------------------------------------------------------------------
# Testbench Module to Predict a Single Image

import sys

sys.path.insert(1, '/Users/elton/diagnosis/src/lib')

from lib import *
from tf_lib import *

img   = cv2.imread('pacient.png')
img1  = np.array(img).astype('float32')/255
img2  = transform.resize(img1, (150, 150, 3))
img3  = np.expand_dims(img2, axis=0)

model = load_model('xception.h5')

r = model.predict(img3)

scores = r
print(scores)

font = {'family': 'Courier New', 'size': 10}

plt.rc('font', **font)

plt.title("COVID-19: "            + str(round(scores[0][0]*100,1)) + "%" + "\n" + 
	      "NORMAL: "              + str(round(scores[0][1]*100,1)) + "%" + "\n" +
	      "PNEUMONIA BACTERIAL: " + str(round(scores[0][2]*100,1)) + "%" + "\n" +
	      "PNEUMONIA VIRAL: "     + str(round(scores[0][3]*100,1)) + "%")

plt.imshow(img2)
plt.show()