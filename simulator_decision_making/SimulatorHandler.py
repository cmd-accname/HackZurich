import argparse
import base64
import cv2
import Decision_Making
import time

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]

    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    open_cv_image = np.array(image) 
    #image_array = np.asarray(image)
    decision, result_image = Decision_Making.process_frame(open_cv_image,0.01,30)
    cv2.imwrite('img' + str(round(time.time())) + '.png', result_image)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # image_array = image_array[40:130]
    # image_array = cv2.resize(image_array, (0,0), fx=0.5, fy=0.5)
    
    steering_angle = float(0.15)
    
    if decision == 'right':
        steering_angle = float(0.15)
    elif decision == 'left':
        steering_angle = float(-0.15)
    
    # transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if abs(steering_angle) > 0.1 and speed > 10:
        throttle = 0.0
    else:
        throttle = 0.15
    #throttle = 0.2
    #steering_angle = round(steering_angle,1)
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)