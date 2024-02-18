from serial import Serial
import math
import cv2
import numpy as np
from gpiozero import InputDevice, OutputDevice, PWMOutputDevice
from time import sleep, time
from scipy.special import expit
from rpi_ws281x import *
import argparse

# Definitions for Ports
v_motor = PWMOutputDevice(14)

# LED strip configuration:
LED_COUNT      = 70     # Number of LED pixels.
LED_PIN        = 18      # GPIO pin connected to the pixels (18 uses PWM!).
LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10      # DMA channel to use for generating a signal (try 10)
LED_BRIGHTNESS = 65      # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53
wait_ms = 50

# Constants for distance calculation
FACE_WIDTH = 0.16  # meters
FOCAL_LENGTH = 800  # pixels

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def estimate_distance(pixel_width):
    """Estimates the distance to an object based on its apparent size in the image."""
    return (FACE_WIDTH * FOCAL_LENGTH) / pixel_width

def trigger_condition(distance, velocity):
    if distance < 5 and velocity > 0.5:
        return True
    else: return False

def calc_vibration(distance, velocity):
    if trigger_condition(distance, velocity):
        return 1 - np.tanh(distance)
    else:
        return 0

# def light_feedback(distance, velocity, distance_from_center, obj_ratio, vech_type):
#     if vech_type == "motorcycle":
#         color = Color(255, 0, 0)
#     elif vech_type == "truck":
#         color = Color(0, 255, 0)
#     else:
#         color = Color(0, 0, 255)
#     dlight_norm = distance_from_center % 35
#     obj_light = (int)(obj_ratio * 70)
#     start_obj = dlight_norm - obj_light/2
#     end_obj = dlight_norm + obj_light
#     for i in range(start_obj, end_obj):
#         strip.setPixelColor(i, color)
#         strip.show()
#         time.sleep(wait_ms/1000.0)

# Initialize variables for velocity calculation
previous_distance = None
previous_time = None

# Process arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--clear', action='store_true', help='clear the display on exit')
# args = parser.parse_args()

# # Create NeoPixel object with appropriate configuration.
# strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
# # Intialize the library (must be called once before other functions).
# strip.begin()

# print ('Press Ctrl-C to quit.')
# if not args.clear:
#     print('Use "-c" argument to clear LEDs on exit')

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
            distance_diff = previous_distance - current_distance
            velocity = distance_diff / time_diff  # m/s
            cv2.putText(frame, f"{velocity:.2f} m/s", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Print how far off the center the face is
            center_x = frame.shape[1] / 2
            # 960 pixels in this case
            face_center_x = x + w / 2
            distance_from_center = face_center_x - center_x
            off_center = distance_from_center / center_x # -1 to 1 left is -1 right is 1
            cv2.putText(frame, f"{distance_from_center:.2f} pixels", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        previous_distance = current_distance
        previous_time = current_time
        
        input_val = [previous_time, previous_distance, velocity, distance_from_center]

        v_motor.value = calc_vibration(previous_distance)
        obj_ratio = largest_face.shape[2]/frame.shape[2]

    # Display the resulting frame
    cv2.imshow('Detection, Distance, and velocity Estimation', frame)

    # Break the loop with the 'ESC' key
    if cv2.waitKey(1) == 27:
        break
    #return bbs location from center of the frame , velocity and distance
# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
