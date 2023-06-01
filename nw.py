import cv2 as cv


#img = cv.imread('Untitled.png')
img = cv.imread('t-shirt-army-green.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find contours in the image
contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
image_with_contours = cv.drawContours(img, contours, -1, (0, 255, 0), 3)

# Display the original image with contours
cv.imshow('Image with Contours', image_with_contours)
cv.waitKey(0)
cv.destroyAllWindows()


#print (img2)