# ----------------------------------------------------------------------------
# Testbench Module to Predict a Single Image

import sys

""" If you want to run this Neural Network at your local machine, please, 
    replace the directory described below: """

sys.path.insert(1, '/Users/elton/Documents/FAPESQ/InceptionV3')

from lib import *

img   = cv2.imread('pacient.jpg')
img1  = np.array(img).astype('float32')/255
img2  = transform.resize(img1, (150, 150, 3))
img3  = np.expand_dims(img2, axis=0)

model = load_model('model_v4.h5')

r = model.predict(img3)

#print(model.summary())  

scores = r
print(scores)

font = {'family': 'Courier New', 'size': 10}

plt.rc('font', **font)

plt.title("PNEUMONIA BACTERIAL: "   + str(round(scores[0][0]*100,1)) + "%" + "\n" + 
	      "COVID-19: "              + str(round(scores[0][1]*100,1)) + "%" + "\n" +
	      "EMPHYSEMA: "             + str(round(scores[0][2]*100,1)) + "%" + "\n" +
	      "NORMAL: "                + str(round(scores[0][3]*100,1)) + "%" + "\n" +
	      "PNEUMONIA VIRAL: "       + str(round(scores[0][4]*100,1)) + "%")

plt.imshow(img2)
plt.show()