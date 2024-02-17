import serial
import math

serial_port = 'COM7'
baud_rate = 9600

ser = serial.Serial(serial_port, baud_rate, timeout = 1)

try:
    while True:
        # Insert camera readings
        input_val = 0
        ser.write(input_val.encode())

except KeyboardInterrupt:
    print("Closing serial port")
    ser.close()