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


##########################################################################################3




img1 = cv2.imread('captured_image.jpg')
img2 = cv2.imread('t-shirt-sea-green.jpg')

img_2_shape = img2.shape
roi = img1[0:img_2_shape[0], 0:img_2_shape[1]]
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)

# black-out the area of t-shirt in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
print(img1.shape, mask.shape)

# Take only region of t-shirt from t-shirt image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

# Put t-shirt in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:img_2_shape[0], 0:img_2_shape[1]] = dst


#Create resizable windows for our display images
#cv2.namedWindow('img1_bg', cv2.WINDOW_NORMAL)
#cv2.namedWindow('img2_fg', cv2.WINDOW_NORMAL)
#cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
#cv2.namedWindow('maskinv', cv2.WINDOW_NORMAL)
cv2.namedWindow('res', cv2.WINDOW_NORMAL)
#cv2.namedWindow('img2gray', cv2.WINDOW_NORMAL)
#cv2.namedWindow('nw', cv2.WINDOW_NORMAL)
#cv2.imshow('mask', mask)
#cv2.imshow('maskinv', mask_inv)
#cv2.imshow('img1_bg', img1_bg)
#cv2.imshow('img2_fg', img2_fg)
cv2.imshow('res', img1)
#cv2.imshow('img2gray', img2gray)
#cv2.imshow('nw', img_2_shape)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
