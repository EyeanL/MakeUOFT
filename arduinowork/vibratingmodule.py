import serial

serial_port = 'COM7'
baud_rate = 9600

ser = serial.Serial(serial_port, baud_rate, timeout = 1)