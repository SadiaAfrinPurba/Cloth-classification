from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage import exposure
from skimage import feature
from imutils import paths
import numpy as np
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2
from PIL import Image #needs because of displaying the images in GUI window
from PIL import ImageTk
from tkinter import Tk, Label, filedialog, Button
#setup
app = argparse.ArgumentParser()
app.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
app.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(app.parse_args())

print ("[INFO] extracting features by using HOG...")
data = []
labels = []

for imagePath in paths.list_images(args["training"]):
	
	make = imagePath.split("/")[-2]
	image = cv2.imread(imagePath)
	#cv2.imshow("Original Image", image)
	imgUMat = image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Gray Image", gray)
	(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	edged = cv2.Canny(gray,50, 100)
	cv2.imshow("After canny edge detection", edged)
	edgedBW = cv2.Canny(im_bw,50, 100)
	logo = cv2.resize(edged, (200, 100))
 

	(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualise=True)
 

	data.append(H)
	labels.append(make)
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 256))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Image", hogImage)


	
	image = cv2.resize(image, (200, 100))
	cv2.putText(image, str(labels), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 2)
	cv2.imshow("Image", image)


# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print("[INFO] evaluating...")
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	logo = cv2.resize(gray, (200, 100))

	
	
	(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
	pred = model.predict(H.reshape(1, -1))[0]

	# score = model.predict(H.reshape(1, -1))[1]
	# accuracy = accuracy_score(make, score)
	# print(accuracy)

	
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 256))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Image #{}".format(i + 1), hogImage)


	
	image = cv2.resize(image, (200, 100))
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)

	cv2.imshow("Test Image #{}".format(i + 1), image)
	k = cv2.waitKey(5000)
