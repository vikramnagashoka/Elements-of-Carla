#!/usr/bin/env python

import eventlet
eventlet.monkey_patch(socket=True, select=True, time=True)

import eventlet.wsgi
import socketio
from flask import Flask

from bridge import Bridge
from conf import conf

sio = socketio.Server()
app = Flask(__name__)
msgs = []

dbw_enable = False
image_counter = 0

@sio.on('connect')
def connect(sid, _environ):
    print("connect ", sid)

def send(topic, data):
    msgs.append((topic, data))
    #sio.emit(topic, data=json.dumps(data), skip_sid=True)

bridge = Bridge(conf, send)

@sio.on('telemetry')
def telemetry(_sid, data):
    global dbw_enable
    if data["dbw_enable"] != dbw_enable:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
    bridge.publish_odometry(data)
    for _ in range(len(msgs)):
        topic, data = msgs.pop(0)
        sio.emit(topic, data=data, skip_sid=True)

@sio.on('control')
def control(_sid, data):
    bridge.publish_controls(data)

# @sio.on('obstacle')
# def obstacle(sid, data):
#     bridge.publish_obstacles(data)

# @sio.on('lidar')
# def obstacle(_sid, data):
#     bridge.publish_lidar(data)

@sio.on('trafficlights')
def trafficlights(_sid, data):
    bridge.publish_traffic(data)

@sio.on('image')
def image(_sid, data):
    global image_counter
    image_counter += 1
    if image_counter == 2:
        image_counter = 0
        bridge.publish_camera(data)


if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
