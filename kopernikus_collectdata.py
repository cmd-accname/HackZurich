

from kopernikus_middleware import *

#usb_cam, onboard_cam = get_image()
cam = get_image(num=1, onboard=False, all_cam=False)
motors = init_actuator()
remote = init_remote()

# Define your class
class Save_data:
    def run(self,frame, commands):
        """
        <TODO: Write your code to save data.>
        """
        pass # throttle, brake, steering

# export the model
model = Save_data()

import time
time.sleep(10)
while(True):

	#Capture frame-by-frame
   	ret, frame = cam.read()
    # Get throttle(commands[0]) and steering(commands[1]) values
    commands = drive_remote(motors, remote)
    # save data
    model.run(frame, commands)
