import cv2
import mediapipe as mp


# Define the coordinates of the point you want to track
POINT_INDEX = 12 # This represents the right elbow
POINT_INDEX2 = 23

# Initialize Mediapipe Pose Detection library
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the image to overlay
img = cv2.imread('t-shirt-sea-green.jpg')

img_shape = img.shape
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)

img_fg = cv2.bitwise_and(img, img, mask=mask_inv)


# Initialize the webcam
cap = cv2.VideoCapture(0)


#######################################################
# Start the pose detection pipeline
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        # Read a frame from the webcam
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (1280, 720))

        # Convert the image to RGB for processing
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make the pose detection
        results = pose.process(image)

        image_height, image_width, _ = image.shape

        # Draw the landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the coordinates of the point of interest
        if results.pose_landmarks:
            x = results.pose_landmarks.landmark[POINT_INDEX].x
            y = results.pose_landmarks.landmark[POINT_INDEX].y

            xn = results.pose_landmarks.landmark[POINT_INDEX2].x
            yn = results.pose_landmarks.landmark[POINT_INDEX2].y

            x_pixels = int(x * image_width)
            y_pixels = int(y * image_height)

            x_pixels2 = int(xn * image_width)
            y_pixels2 = int(yn * image_height)

            # Print the coordinates to the console
            print(f"X: {x_pixels}, Y: {y_pixels}")

            # Define the region of interest (ROI) coordinates
            x1, y1 = x_pixels, y_pixels  # top left corner
            # x2, y2 = x1 + img_fg.shape[1], y1 + img_fg.shape[0]  # bottom right corner
            x2, y2 = x_pixels2, y_pixels2

        
            # Extract the ROI from the webcam frame
            roi = image[y1-50:y2, x1-50:x2+100]

            # Resize the overlay image to match the size of the ROI
            resized_overlay = cv2.resize(img_fg, (roi.shape[1], roi.shape[0]))

            # Overlay the image onto the ROI
            result = cv2.addWeighted(roi, 0, resized_overlay, 1, 0)

            # Replace the ROI in the original frame
            image[y1-50:y2, x1-50:x2+100] = result

            # Display the resulting frame
            cv2.imshow('Webcam', image)


        if cv2.waitKey(1) == ord('q'):
            break


######################################################
"""
# Define the region of interest (ROI) coordinates
x1, y1 = x, y # top left corner
x2, y2 = x1 + img_fg.shape[1], y1 + img_fg.shape[0] # bottom right corner



while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Extract the ROI from the webcam frame
        roi = frame[y1:y2, x1:x2]

        # Resize the overlay image to match the size of the ROI
        resized_overlay = cv2.resize(img_fg, (roi.shape[1], roi.shape[0]))

        # Overlay the image onto the ROI
        result = cv2.addWeighted(roi, 0, resized_overlay, 1, 0)


        # Replace the ROI in the original frame
        frame[y1:y2, x1:x2] = result

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
"""

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()