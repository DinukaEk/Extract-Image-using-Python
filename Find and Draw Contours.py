import numpy as np
import cv2


img = cv2.imread('t-shirt-army-green.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)


# create canvas
canvas = np.zeros(img.shape, np.uint8)
canvas.fill(255)

# create background mask
mask = np.zeros(img.shape, np.uint8)
mask.fill(255)

# create new background
new_background = np.zeros(img.shape, np.uint8)
new_background.fill(255)


contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours_mask, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print("Number of contours = " + str(len(contours)))
print(contours[0])

#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)


# contours traversal
for contour in range(len(contours)):
    # draw current contour

    #cv2.drawContours(canvas, contours, contour, (0, 0, 0), 3)

    if contour == 1:
        cv2.drawContours(canvas, contours, contour, (0, 0, 0), 3)

bitwise_OR = cv2.bitwise_or(canvas, img)
# most significant contours traversal
for contour in range(len(contours_mask)):
    # create mask
    if contour != 1:
        cv2.fillConvexPoly(mask, contours_mask[contour], (0, 0, 0))

    # create background
    if contour != 1:
      cv2.fillConvexPoly(new_background, contours_mask[contour], (0, 255, 0))




# Display the result
#cv2.imshow('Image', img)
#cv2.imshow('Image GRAY', imgray)
cv2.imshow('Contours', canvas)
#cv2.imshow('OR', bitwise_OR)
#cv2.imshow('Background mask', mask)
#cv2.imshow('New background', new_background)
#cv2.imshow('Output', cv2.bitwise_and(img, new_background))

cv2.waitKey(0)
cv2.destroyAllWindows()