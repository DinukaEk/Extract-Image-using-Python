import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('t-shirt-army-green.jpg')

image1_copy = img.copy()
gray_image = cv2.cvtColor(image1_copy, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#contours_mask, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#print("Number of contours = " + str(len(contours)))
#print(contours[0])


# Find all contours in the image.
contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Draw the selected contour
cv2.drawContours(gray_image, contours, -1, (0, 255, 0), 3);




# Find all contours in the image.
#contours, hierarchy = cv2.findContours(imgray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Retreive the biggest contour
#biggest_contour = max(contours, key = cv2.contourArea)

# Draw the biggest contour
#cv2.drawContours(image1_copy, biggest_contour, -1, (0, 255, 0), 4);

# Display the results
plt.figure(figsize=[10, 10])
plt.imshow(image1_copy[:, :, ::-1]); plt.axis("off");
cv2.imshow('Contour', image1_copy[:, :, ::-1])


cv2.waitKey(0)
cv2.destroyAllWindows()