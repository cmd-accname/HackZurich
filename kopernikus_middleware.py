import numpy as np
import cv2
import serial
import pygame
import time
import struct




def get_image(num=1, onboard=False, all_cam=False):
    # Start the video feed from USB camera

    if onboard and all_cam:
        # Start video feed from onboard camera
        cap = cv2.VideoCapture(num)
        cap_onboard = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
        return cap, cap_onboard
    elif onboard:
        cap_onboard = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
        return cap_onboard
    else:
        cap = cv2.VideoCapture(num)
        return cap

def init_actuator():
    # Open Serial Connection to Arduino Board
    ser = serial.Serial('/dev/ttyACM0', 9600)
    return ser

def drive_actuator(ser, commands, j):
	thr = int(commands[0]*50 + 50)
	ster = abs(int(commands[1]*50 + 50))
	pygame.event.pump()
	# deadman switch for safety
	if j.get_button(4):
		ser.write(b't')
		time.sleep(0.011)
		ser.write(struct.pack('>B', thr))
		time.sleep(0.011)
		ser.write(struct.pack('>B', ster))
		time.sleep(0.011)
		print(thr, ster)

def init_remote():
    # initiate pygame
    pygame.init()
    j = pygame.joystick.Joystick(0)
    j.init()
    return j

def drive_remote(ser, j):

    pygame.event.pump()

	thr = int(j.get_axis(1)*(-50) + 50)
	ster = abs(int(j.get_axis(2)*50 + 50))
	ser.write(b't')
	time.sleep(0.011)
	ser.write(struct.pack('>B', thr))
	time.sleep(0.011)
	ser.write(struct.pack('>B', ster))
	time.sleep(0.011)
	#return data
	#print (thr, ster)
	return [j.get_axis(1), j.get_axis(2)]
