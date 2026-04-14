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
endpoint = "tcp://localhost:7000"
topic = "SKEL"
socket = None
thread = None

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")



def remove_shm_from_resource_tracker(name):
    rt.unregister(f"/{name}", "shared_memory")





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
    dcc.Graph(id="graph"),
    dcc.Interval(
            id='interval-component',
            interval=50, # in milliseconds
            n_intervals=0)], 
    id = "change-height", 
    style={'display': 'inline-block', 'width': '100%', 'height': '100%'})



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
        socketio.sleep(0.02)  # ~60 FPS



"""# Visualize real-time scatter data
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
        socketio.sleep(0.01)  # ~60 FPS"""



scene = dict(
        xaxis = dict(nticks=10, range=[-2, 2],),
        yaxis = dict(nticks=10, range=[-2, 2],),
        zaxis = dict(nticks=10, range=[-2, 2],)
        )
camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.1),
        eye=dict(x=1.5, y=1.5, z=0.1)
        )

radius=1.0
num_points=200
theta = np.linspace(0, 2 * np.pi, num_points)
xx = radius * np.cos(theta)
yy = radius * np.sin(theta)
ii=0

@dash_app.callback(Output("graph", "figure"), Input('interval-component', 'n_intervals'))
# @app.callback(Output("graph", "figure"), Input('interval-component', 'n_intervals'))
def background_task(n_intervals):
    global ii

    t0 = time.time()
    topic, message = socket.recv_string().split(" ", 1)
    array = json.loads(message)
    data = array

    print(array)

    # x = []
    # y = []
    # z = []
    # for [*pnt] in data:
    #     x.append(pnt[0])
    #     y.append(pnt[1])
    #     z.append(pnt[2])

    if ii ==200:
        ii = 0
    ii+=1
    x=[xx[ii], xx[ii], xx[ii], xx[ii], xx[ii], xx[ii], xx[ii], xx[ii], xx[ii], xx[ii]]
    y=[yy[ii], yy[ii], yy[ii], yy[ii], yy[ii], yy[ii], yy[ii], yy[ii], yy[ii], yy[ii]]
    z=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])

    # for (a, b) in EDGES:
    #     fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='lines')

    fig.update_layout(scene=scene, scene_camera=camera, scene_aspectmode='cube', height=1200, width=1500, margin=dict(r=20, l=20, b=10, t=10))

    t1 = time.time()
    print(f"\rTime: {t1-t0}", end="")

    time.sleep(0.3)
    return fig



"""@socketio.on('connect')
def handle_connect():
    # socketio.start_background_task(background_task)
    socketio.start_background_task(stream)"""



@app.route("/")
def index():
    return render_template("test_3.html")



def main():
    global socket, topic
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(endpoint)

    webbrowser.open_new('http://127.0.0.1:5000/')

    # socketio.start_background_task(background_task)
    socketio.start_background_task(stream)
    socketio.run(app, host="127.0.0.1", port=5000)



if __name__ == "__main__":
    main()