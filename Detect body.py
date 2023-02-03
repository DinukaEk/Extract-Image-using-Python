import cv2

# Load image
img = cv2.imread("Man without tshirt 2.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a HOG descriptor
hog = cv2.HOGDescriptor()

# Set the SVM detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect people in the image
boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.05)

# Draw bounding boxes around detected people
for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Show the image with bounding boxes
cv2.imshow("Human Body Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
