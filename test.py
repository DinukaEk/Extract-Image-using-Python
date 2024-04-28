import cv2
import mediapipe as mp


def capture_image():
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Failed to open webcam")
        return

    while True:
        ret, frame = webcam.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1024, 768))

        cv2.imshow("Webcam", frame)

        # Check for keyboard input
        key = cv2.waitKey(1)

        # If 'c' is pressed, capture and save the image
        if key == ord('c'):
            image_name = "captured_image.jpg"
            cv2.imwrite(image_name, frame)
            break

        # If 'q' is pressed, quit the program
        if key == ord('q'):
            break

    # Release the webcam and close all windows
    webcam.release()
    cv2.destroyAllWindows()


# Call the capture_image function
capture_image()


#################################################################################


# Initialize Mediapipe pose detection model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the input image
image = cv2.imread('captured_image.jpg')
#image = cv2.imread('t-shirt-sea-green.jpg')
results = pose.process(image)
landmarks = results.pose_landmarks
image_height, image_width, _ = image.shape


# Convert the image to RGB color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect pose from the image
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    results = pose.process(image)

    # extract the coordinates of the first landmark (nose)
    x = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
    y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

    x_pixels = int(x * image_width)
    y_pixels = int(y * image_height)

    print(f"Coordinates: ({x_pixels}, {y_pixels})")

    # Draw the detected pose on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the image with the detected pose
    cv2.imshow("Pose Detection", image)


##########################################################################################3




img1 = cv2.imread('captured_image.jpg')
img2 = cv2.imread('t-shirt-sea-green.jpg')

# Define the coordinates where you want to place img2
x1 = 100  # x-coordinate of the top-left corner
y1 = 150  # y-coordinate of the top-left corner

#x1, y1 = x_pixels, y_pixels  # top left corner
#x2, y2 = x1 + img2.shape[1], y1 + img2.shape[0]  # bottom right corner

# Get the shape of img2
img2_shape = img2.shape

# Calculate the region of interest in img1 based on img2's shape and given coordinates
roi = img1[y1:y1+img2_shape[0], x1:x1+img2_shape[1]]

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
img1[y1:y1+img2_shape[0], x1:x1+img2_shape[1]] = dst

cv2.imshow('res', img1)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
