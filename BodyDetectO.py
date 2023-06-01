import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Webcam input
cap = cv2.VideoCapture(0)

LANDMARK_INDEX = mp_pose.PoseLandmark.RIGHT_SHOULDER

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        image.flags.writeable=False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image_height, image_width, _ = image.shape

        #Draw pose annotation on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and results.pose_landmarks.landmark[LANDMARK_INDEX]:
            landmark = results.pose_landmarks.landmark[LANDMARK_INDEX]
            x = landmark.x
            y = landmark.y

            arm_x_pixels = int(x * image_width)
            arm_y_pixels = int(y * image_height)

            #print(f"Coordinates of arm: ({arm_x_pixels}, {arm_y_pixels})")



        annotated_image = image.copy()
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        #Flip image
        #cv2.imshow('Pose',cv2.flip(annotated_image, 1))
        cv2.imshow('Image',cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()