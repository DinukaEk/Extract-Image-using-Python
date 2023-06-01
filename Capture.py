import cv2


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
