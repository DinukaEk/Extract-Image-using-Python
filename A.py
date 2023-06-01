import cv2
import mediapipe as mp

# Define the coordinates of the point you want to track
POINT_INDEX = 12 # This represents the right elbow

# Initialize Mediapipe Pose Detection library
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Start the pose detection pipeline
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        # Read a frame from the webcam
        ret, image = cap.read()
        image = cv2.flip(image, 1)

        # Convert the image to RGB for processing
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make the pose detection
        results = pose.process(image)

        # Draw the landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the coordinates of the point of interest
        if results.pose_landmarks:
            x = results.pose_landmarks.landmark[POINT_INDEX].x
            y = results.pose_landmarks.landmark[POINT_INDEX].y

            # Print the coordinates to the console
            print(f"X: {x}, Y: {y}")

        # Show the image
        cv2.imshow("MediaPipe Pose Detection", image)

        # Exit the program if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
