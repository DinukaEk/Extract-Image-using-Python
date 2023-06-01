import cv2
import mediapipe as mp

# Initialize Mediapipe pose detection model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the input image
image = cv2.imread('Man without tshirt 2.jpg')
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
    arm_x = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
    arm_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    arm_z = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z

    arm_x_pixels = int(arm_x * image_width)
    arm_y_pixels = int(arm_y * image_height)

    print(f"Coordinates of arm: ({arm_x_pixels}, {arm_y_pixels})")

    # Draw the detected pose on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the image with the detected pose
    cv2.imshow("Pose Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()