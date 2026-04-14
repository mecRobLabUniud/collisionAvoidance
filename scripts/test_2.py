from flask import Flask, jsonify, render_template, send_from_directory
import json
import plotly.graph_objects as go
import plotly.utils
import plotly.io as pio
import time
import numpy as np
from multiprocessing import shared_memory
from PIL import Image
import numpy as np
import cv2
import multiprocessing.resource_tracker as rt
import time
import base64
import webbrowser
import zmq
import logging
import io
import pyvista as pv
from flask_socketio import SocketIO
from dash import Dash, dcc, html, Input, Output

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

TARGET_KEYPOINTS = list(range(17))  # 0..12 pelvis-up
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]

# Parameters
H, W, C = 480, 848, 3  # adjust to match your image
dtype = np.uint8
endpoint = "tcp://localhost:6000"
topic = "SKEL"
socket = None
thread = None

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")



def remove_shm_from_resource_tracker(name):
    rt.unregister(f"/{name}", "shared_memory")



"""def make_3d_plot(x, y, z, title="3D Plot"):
    global socket

    topic, message = socket.recv_string().split(" ", 1)
    array = json.loads(message)
    data = array

    x = []
    y = []
    z = []
    for [*pnt] in data:
        x.append(pnt[0])
        y.append(pnt[1])
        z.append(pnt[2])

    return fig



def cv2_to_b64(img):
    is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not is_success:
        return None
    encoded = base64.b64encode(buffer).decode("utf-8")
    return encoded # "data:image/jpeg;base64," + encoded



@app.route("/image")
def stream():
    # Read image data from shared memory
    shm = shared_memory.SharedMemory(name="shared_image")
    remove_shm_from_resource_tracker(shm.name)

    arr = np.ndarray((H, W, C), dtype=dtype, buffer=shm.buf)
    img = arr.copy()
    pic = cv2_to_b64(img)
    shm.close()
    return jsonify({"image": pic})



@app.route("/plot")
def scatter3D_plot():
    pass"""



def cv2_to_b64(img):
    is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not is_success:
        return None
    encoded = base64.b64encode(buffer).decode("utf-8")
    return encoded # "data:image/jpeg;base64," + encoded




dash_app = Dash(
    __name__,
    server=app,          # reuse Flask server
    url_base_pathname="/dash/" # Dash will live at /dash/*
)

dash_app.layout = html.Div([
    html.H1("My Dash app inside Flask"),
    dcc.Graph(
        id="example-graph",
        figure={
            "data": [{"x": [1, 2, 3], "y": [4, 2, 5], "type": "bar"}],
            "layout": {"title": "Example"}
        }
    )
])




# Read image data from shared memory
def stream():
    while True:
        shm = shared_memory.SharedMemory(name="shared_image0")
        remove_shm_from_resource_tracker(shm.name)

        arr = np.ndarray((H, W, C), dtype=dtype, buffer=shm.buf)
        img = arr.copy()
        pic = cv2_to_b64(img)
        shm.close()

        socketio.emit('update_stream', {'frame': pic})
        socketio.sleep(0.01)  # ~60 FPS



# Visualize real-time scatter data
def background_task():    
    global socket

    while True:
        topic, message = socket.recv_string().split(" ", 1)
        array = json.loads(message)
        data = array

        x = []
        y = []
        z = []
        for [*pnt] in data:
            if not np.isnan(pnt[0]):
                x.append(pnt[0])
                y.append(pnt[1])
                z.append(pnt[2])

        socketio.emit('update_scatter', {'x': x, 'y': y, 'z': z})
        socketio.sleep(0.01)  # ~60 FPS



@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(background_task)
    socketio.start_background_task(stream)



@app.route("/")
def index():
    return render_template("test_2.html")



def main():
    global socket, topic
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(endpoint)

    webbrowser.open_new('http://127.0.0.1:5000/')

    # socketio.start_background_task(background_task)
    # socketio.start_background_task(stream)
    socketio.run(app, host="127.0.0.1", port=5000)



if __name__ == "__main__":
    main()