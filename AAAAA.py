import cv2

img1 = cv2.imread('Man without tshirt 2.jpg')
img2 = cv2.imread('t-shirt-sea-green.jpg')

# Define the coordinates where you want to place img2
x = 100  # x-coordinate of the top-left corner
y = 150  # y-coordinate of the top-left corner

# Get the shape of img2
img2_shape = img2.shape

# Calculate the region of interest in img1 based on img2's shape and given coordinates
roi = img1[y:y+img2_shape[0], x:x+img2_shape[1]]

# Convert img2 to grayscale
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create a binary mask from img2_gray
ret, mask = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Extract the background of the ROI using the mask
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

# Extract the foreground (img2) using the inverted mask
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

# Combine the background and foreground to get the final result
dst = cv2.add(img1_bg, img2_fg)

# Place the result back into the ROI of img1
img1[y:y+img2_shape[0], x:x+img2_shape[1]] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()