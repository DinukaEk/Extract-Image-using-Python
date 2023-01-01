import cv2
import numpy as np
from matplotlib import pyplot as plt


def display(img, cmap='gray'):
 fig = plt.figure(figsize=(12,10))
 ax = fig.add_subplot(111)
 ax.imshow(img,cmap='gray')


#read in image --> added hyperlink to image
img = cv2.imread("t-shirt-army-green.jpg")

#blur and grayscale img
img = cv2.medianBlur(img, 5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#apply threshold to grayscale image + otsu's method
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#NOISE REMOVAL
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations=6)

#set seeds to distinguish between fore and background
sure_bg = cv2.dilate(opening, kernel, iterations = 2)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)

#foreground
ret, sure_fg = cv2.threshold(dist_transform,0.80*dist_transform.max(),255,0)

#find region between shade and core white colours
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#create label markers for watershed algorithm to make distinction between upper and lower body
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1

#set unknown equal to black
markers[unknown==255] = 0

#apply markers to algorithm
markers = cv2.watershed(img,markers)
image,contours,hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)


# create contours
for i in range(len(contours)):
#external contour
 if hierarchy[0][i][3] == -1:
   cv2.drawContours(img,contours,i,255,-1)