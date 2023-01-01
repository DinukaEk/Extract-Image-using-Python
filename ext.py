import cv2 as cv
import numpy as np

#img = cv.imread('smarties.png')
img = cv.imread('t-shirt-army-green.jpg')
cv.imshow('Original', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

center_coordinates = (144, 233)

radius = 10

mask = cv.circle(blank, center_coordinates, radius, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Output', masked)

cv.waitKey(0)