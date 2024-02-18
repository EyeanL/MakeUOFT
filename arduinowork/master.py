from serial import Serial
import math
import cv2
import time
import numpy as np
# Constants for distance calculation
FACE_WIDTH = 0.16  # meters
FOCAL_LENGTH = 800  # pixels

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def estimate_distance(pixel_width):
    """Estimates the distance to an object based on its apparent size in the image."""
    return (FACE_WIDTH * FOCAL_LENGTH) / pixel_width

# Initialize variables for speed calculation
previous_distance = None
previous_time = None

serial_port = 'COM7'
baud_rate = 9600

ser = Serial(serial_port, baud_rate, timeout = 1)

# Initialize the webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        # Find the largest face based on area (w * h)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        current_distance = estimate_distance(w)
        cv2.putText(frame, f"{current_distance:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if previous_distance is not None and previous_time is not None:
            time_diff = current_time - previous_time
            distance_diff = current_distance - previous_distance
            speed = distance_diff / time_diff  # m/s
            cv2.putText(frame, f"{speed:.2f} m/s", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Print how far off the center the face is
            center_x = frame.shape[1] / 2
            # 960 pixels in this case
            face_center_x = x + w / 2
            distance_from_center = face_center_x - center_x
            off_center = distance_from_center / center_x # -1 to 1 left is -1 right is 1
            cv2.putText(frame, f"{distance_from_center:.2f} pixels", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        previous_distance = current_distance
        previous_time = current_time
        
        input_val = [previous_time, previous_distance, distance_from_center]
        ser.write(input_val.encode())

    # Display the resulting frame
    cv2.imshow('Detection, Distance, and Speed Estimation', frame)

    # Break the loop with the 'ESC' key
    if cv2.waitKey(1) == 27:
        break
    #return bbs location from center of the frame , speed and distance
# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
