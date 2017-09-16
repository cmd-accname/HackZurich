import numpy as np
import cv2
import serial
import pygame
import time
import struct


class Vehicle():
    """docstring for ."""
    def __init__(self):
        self.j = init_remote()
        self.ser = serial.Serial('/dev/ttyACM0', 9600)
        #pass

    def init_remote():
        # initiate pygame
        pygame.init()
        j = pygame.joystick.Joystick(0)
        j.init()
        return j

    def drive_actuator():
    	pygame.event.pump()
    	# deadman switch for safety
    	if self.j.get_button(4):
    		ser.write(b't')
    		time.sleep(0.011)
    		self.ser.write(struct.pack('>B', self.throttle))
    		time.sleep(0.011)
    		self.ser.write(struct.pack('>B', self.steering))
    		time.sleep(0.011)
    		#print(self.throttle, self.steering)

    def status(self):
        pass

    def setGear(self, gear):
        if gear == "GEAR_DIRECTION_FORWARD":
            self.go = 1
        elif gear ==  "GEAR_DIRECTION_BACKWARD":
            self.go = -1
        else:
            self.go = 0

    def setSteeringAngle(self, angle):
        self.steering = abs(int((angle * 50) + 50))
        drive_actuator()
        #print(self.steering)

    def setThrottle(self, position):
        self.throttle = int((self.go * position * 50) + 50)
        drive_actuator()
        #print(self.throttle)
