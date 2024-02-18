import cv2
import time
import numpy as np

# Constants for distance calculation
FACE_WIDTH = 0.16  # meters for face
CAR_WIDTH = 1.8  # meters for an average car width
TWO_WHEELER_WIDTH = 0.5  # meters for an average two-wheeler width
BUS_WIDTH = 2.5  # meters for an average bus width
PEDESTRIAN_WIDTH = 0.5  # meters for an average pedestrian width
FOCAL_LENGTH = 800  # pixels

# Load the cascade classifiers for face and car detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cars_cascade = cv2.CascadeClassifier('cars.xml')
pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')
two_wheeler_cascade = cv2.CascadeClassifier('two_wheeler.xml')
bus_cascade = cv2.CascadeClassifier('Bus_front.xml')
def estimate_distance(pixel_width, object_type):
    """Estimates the distance to an object based on its apparent size in the image."""
    if object_type == "face":
        return (FACE_WIDTH * FOCAL_LENGTH) / pixel_width
    elif object_type == "car":
        return (CAR_WIDTH * FOCAL_LENGTH) / pixel_width
    elif object_type == "pedestrian":
        return (PEDESTRIAN_WIDTH * FOCAL_LENGTH) / pixel_width
    elif object_type == "two-wheeler":
        return (TWO_WHEELER_WIDTH * FOCAL_LENGTH) / pixel_width
    elif object_type == "bus":
        return (BUS_WIDTH * FOCAL_LENGTH) / pixel_width
# Initialize variables for speed calculation
previous_distance_face = None
previous_time_face = None
previous_distance_car = None
previous_time_car = None

# Initialize the webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    # Detect cars
    #cars = cars_cascade.detectMultiScale(gray, 1.1, 2)

    # Process faces for distance and speed estimation
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.1, 4)
    two_wheelers = two_wheeler_cascade.detectMultiScale(gray, 1.1, 4)
    buses = bus_cascade.detectMultiScale(gray, 1.1, 4)
    previous_distance = None
    previous_time = None
    if len(faces) > 0:
        # Find the largest face based on area (w * h)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        current_distance = estimate_distance(w, "face")
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

    # Process cars for distance and speed estimation
    # for (x, y, w, h) in cars:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     current_distance_car = estimate_distance(w, "car")
    #     cv2.putText(frame, f"{current_distance_car:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     if previous_distance_car is not None and previous_time_car is not None:
    #         time_diff_car = current_time - previous_time_car
    #         distance_diff_car = current_distance_car - previous_distance_car
    #         speed_car = distance_diff_car / time_diff_car  # m/s
    #         cv2.putText(frame, f"{speed_car:.2f} m/s", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     previous_distance_car = current_distance_car
    #     previous_time_car = current_time
    previous_distance_pedestrian = None
    previous_time_pedestrian = None
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        current_distance_pedestrian = estimate_distance(w, "pedestrian")
        cv2.putText(frame, f"{current_distance_pedestrian:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if previous_distance_pedestrian is not None and previous_time_pedestrian is not None:
            time_diff_pedestrian = current_time - previous_time_pedestrian
            distance_diff_pedestrian = current_distance_pedestrian - previous_distance_pedestrian
            speed_pedestrian = distance_diff_pedestrian / time_diff_pedestrian
            cv2.putText(frame, f"{speed_pedestrian:.2f} m/s", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        previous_distance_pedestrian = current_distance_pedestrian
        previous_time_pedestrian = current_time
    previous_distance_two_wheeler = None
    previous_time_two_wheeler = None
    for (x, y, w, h) in two_wheelers:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        current_distance_two_wheeler = estimate_distance(w, "two-wheeler")
        cv2.putText(frame, f"{current_distance_two_wheeler:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if previous_distance_two_wheeler is not None and previous_time_two_wheeler is not None:
            time_diff_two_wheeler = current_time - previous_time_two_wheeler
            distance_diff_two_wheeler = current_distance_two_wheeler - previous_distance_two_wheeler
            speed_two_wheeler = distance_diff_two_wheeler / time_diff_two_wheeler
            cv2.putText(frame, f"{speed_two_wheeler:.2f} m/s", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        previous_distance_two_wheeler = current_distance_two_wheeler
        previous_time_two_wheeler = current_time
    previous_distance_bus = None
    previous_time_bus = None
    for (x, y, w, h) in buses:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        current_distance_bus = estimate_distance(w, "bus")
        cv2.putText(frame, f"{current_distance_bus:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        if previous_distance_bus is not None and previous_time_bus is not None:
            time_diff_bus = current_time - previous_time_bus
            distance_diff_bus = current_distance_bus - previous_distance_bus
            speed_bus = distance_diff_bus / time_diff_bus
            cv2.putText(frame, f"{speed_bus:.2f} m/s", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        previous_distance_bus = current_distance_bus
        previous_time_bus = current_time
    
   
    # Display the resulting frame
    cv2.imshow('Detection, Distance, and Speed Estimation', frame)

    # Break the loop with the 'ESC' key
    if cv2.waitKey(1) == 27:
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
