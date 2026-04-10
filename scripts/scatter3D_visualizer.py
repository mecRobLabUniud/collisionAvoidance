#!/usr/bin/env python3

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
import cv2
import zmq
import json
import signal
import threading
import time
import base64
from multiprocessing import shared_memory
from PIL import Image
import numpy as np
import cv2
import multiprocessing.resource_tracker as rt
import time

TARGET_KEYPOINTS = list(range(17))  # 0..12 pelvis-up
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]


# Parameters
endpoint = "tcp://localhost:6000"
topic = "SKEL"
running = True
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

app = Dash(__name__)

data = None
pic = None
socket = None

def remove_shm_from_resource_tracker(name):
    rt.unregister(f"/{name}", "shared_memory")

# Must know shape/dtype in advance (or pass via a side channel)
H, W, C = 480, 848, 3  # adjust to match your image
dtype = np.uint8



def cv2_to_b64(img):    
    """Convert OpenCV image to base64 data URI."""
    is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not is_success: return None
    encoded = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + encoded








"""class SkeletonVisualizer:
    def __init__(self, socket):
        self.started = False
        self.socket = socket
        self.data = None
        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self.data_receiver, args=())

    def start(self):
        if self.started:
            return
        self.started = True
        self.thread.start()
        return self

    def data_receiver(self):
        global running, data, pic
        while running:
            topic, message = self.socket.recv_string().split(" ", 1)
            array = json.loads(message)
            # print(f"Received: {array}")
            with self.mutex:
                self.data = array
                data = array
        quit()

    def read_frame(self):
        with self.mutex:
            frame = self.data.copy() if self.data is not None else None
        return frame
    
    def load_interface(self):
        app.run(debug=True)"""






@app.callback([Output("graph", "figure"), Output("dynamic-img", "src")], Input('interval-component', 'n_intervals'))
# @app.callback(Output("graph", "figure"), Input('interval-component', 'n_intervals'))
def update_bar_chart(n_intervals):
    # global data, pic
    t1 = time.time()

    # Read image data from shared memory
    shm = shared_memory.SharedMemory(name="shared_image1")
    # remove_shm_from_resource_tracker(shm.name) 
    arr = np.ndarray((H, W, C), dtype=dtype, buffer=shm.buf)
    img = arr.copy()
    pic = cv2_to_b64(img)
    shm.close() 

    t2 = time.time()

    global socket
    topic, message = socket.recv_string().split(" ", 1)
    data = json.loads(message)

    print(message)

    x = []
    y = []
    z = []
    for [*pnt] in data:
        x.append(pnt[0])
        y.append(pnt[1])
        z.append(pnt[2])
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])

    for (a, b) in EDGES:
        fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='lines')

    fig.update_layout(scene=scene, scene_camera=camera, scene_aspectmode='cube', height=1200, width=1500, margin=dict(r=20, l=20, b=10, t=10))

    t3 = time.time()
    # print(f"\rTime elapsed for updating figure: {t3 - t2}", end=" ")

    return fig, pic



# Main loop to receive data via ZeroMQ and update the plot
def main():
    global socket
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.connect(endpoint)

    # Subscribe to "news" topic (prefix match)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    app.layout = html.Div([
                html.H1('Skeleton tracking 3D scatter'),
                html.Div([
                    dcc.Graph(id="graph"),
                    html.Img(id="dynamic-img", style={"height": "300px"})],
                    style={"display": "flex", "width": "100%"}),
                dcc.Interval(
                        id='interval-component',
                        interval=30, # in milliseconds
                        n_intervals=0)], 
                id = "change-height", 
                style={'display': 'inline-block', 'width': '100%', 'height': '100%'})

    # Gestione segnali per chiusura pulita (es. CTRL+C o kill da script bash)
    # def signal_handler(sig, frame):
    #     global running
    #     running = False
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)

    # Launch data receiver thread and load the Dash interface
    # vis = SkeletonVisualizer(socket).start()
    #vis.load_interface()

    app.run(debug=True, port=8001)



if __name__ == "__main__":
    main()
    