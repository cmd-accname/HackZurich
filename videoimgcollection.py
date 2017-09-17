"""
This is just an easy class to collect sample images
"""
import cv2
import os
import time

video = True
nthframe = 20


cap = cv2.VideoCapture(1)  # 640 x 480
t = str(time.time())
directory = t
counter = 0
# caputer a video or images?
if video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(t+'-output.avi', fourcc, 60.0, (640, 480))
    counter = nthframe
else:
    counter2 = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    counter += 1
    cv2.imshow('frame', frame)

    if ret and counter > nthframe:
        # create an folder and save the images there
        if not os.path.exists(directory) and not video:
            os.makedirs(directory)
            cv2.imwrite(directory+'/'+str(counter2)+'img.png', frame)
            counter2 += 1
            counter = 0
        else:
            out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
