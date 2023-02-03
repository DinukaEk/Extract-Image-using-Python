import cv2

# Load the pre-trained model
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Read the input image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect human bodies in the image
bodies = body_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected bodies
for (x, y, w, h) in bodies:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show the output image
cv2.imshow('Detected Bodies', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
