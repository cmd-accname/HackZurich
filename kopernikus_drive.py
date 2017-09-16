from kopernikus_middleware import *
import numpy as np
import random

# usb camera
cam = get_image(num=1, onboard=True, all_cam=False)
# initiate serial communication
motors = init_actuator()
# initiate joystick
remote = init_remote()


# Define your model here
class Model:
    def run(self,frame):
        """
        <TODO: Define your model here.>
        <The model return a list with two floats.>
        <[0.0, 0.0] = [throttle, steering]>
        <throttle 1 = fwd max; throttle -1 = rev max; throttle 0 = neutral>
        <steering 1 = right max; steering -1 = left max; steering 0 = center >
        """
        return [0.0, 0.0] # throttle, steering

# import your the model
model = Model()
# delay for serial communication
time.sleep(10)

while(True):

	# Capture frame-by-frame
    ret, frame = cam.read()
	# Run the model
	commands = model.run(frame)
	# Drive motors
	drive_actuator(motors, commands, remote)
  
