from kopernikus_middleware import *
import time
import csv

plugged = False

#usb_cam, onboard_cam = get_image()
cam = get_image(num=1, onboard=False, all_cam=False)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)
if plugged:
	motors = init_actuator()
	remote = init_remote()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
tnow = str(time.time())
out = cv2.VideoWriter(tnow + 'output.avi', fourcc, 30.0, (640, 480))

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
time.sleep(1)
t = 0
with open(tnow+"test.csv", "w") as csv_file:
	writer = csv.writer(csv_file)

	while(True):
		# Capture frame-by-frame
		ret, frame = cam.read()
		# Get throttle(commands[0]) and steering(commands[1]) values
		if plugged:
			commands = drive_remote(motors, remote)
			model.run(frame, commands)
		if ret:
			# cv2.imwrite(tnow+'-'+str(t)+'.png', frame)
			out.write(frame)
			if plugged:
				print([t]+commands)
				writer.writerow([t]+commands)
			cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
	    			break
		else:
			break

		t += 1

cam.release()
out.release()
