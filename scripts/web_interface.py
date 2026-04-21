#!/usr/bin/env python3

"""
░█░█░█▀▀░█▀▄░░░▀█▀░█▀█░▀█▀░█▀▀░█▀▄░█▀▀░█▀█░█▀▀░█▀▀
░█▄█░█▀▀░█▀▄░░░░█░░█░█░░█░░█▀▀░█▀▄░█▀▀░█▀█░█░░░█▀▀
░▀░▀░▀▀▀░▀▀░░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀░░░▀░▀░▀▀▀░▀▀▀


"""

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
import webbrowser
from statistics import mean
from utils.kalman_filter import KalmanFilter 
# from utils.new_kalman_filter import KalmanFilter 

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
camera = dict(up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.1),
        eye=dict(x=0, y=2, z=0.5))
data = None
pic = None
interfaces = None
H, W, C = 480, 848, 3 
dtype = np.uint8

# Launching Dash app
app = Dash(__name__)



# Unregister shared_memory folder
def remove_shm_from_resource_tracker(name):
    rt.unregister(f"/{name}", "shared_memory")



# Convert OpenCV image to base64 data URI
def cv2_to_b64(img):
    is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not is_success: 
        return None
    encoded = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + encoded



# Starting thread for data acquisition from camera_stream
class SkeletonVisualizer:
    def __init__(self, n: int):
        zctx = zmq.Context.instance()
        socket = zctx.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, f"{topic}_{n}")
        socket.connect(endpoint)
        self.socket = socket

        self.n_device = n
        self.started = False
        self.skeleton_data = None
        self.frame_data = None

    def start(self):
        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self.data_receiver, args=(self.n_device,))
        if self.started:
            return
        self.started = True
        self.thread.start()
        return self

    def data_receiver(self, n: int):
        global running
        while running:
            _, _, message = self.socket.recv_string().split(" ", 2)
            skeleton = json.loads(message)
            with self.mutex:
                self.skeleton_data = skeleton

            shm = shared_memory.SharedMemory(name=f"shared_image_{n}")
            remove_shm_from_resource_tracker(shm.name) 
            arr = np.ndarray((H, W, C), dtype=dtype, buffer=shm.buf)
            img = arr.copy()
            frame = cv2_to_b64(img)
            shm.close() 
            with self.mutex:
                self.frame_data = frame

    def read_skeleton(self):
        with self.mutex:
            skeleton = self.skeleton_data.copy() if self.skeleton_data is not None else None
        return skeleton
    
    def read_frame(self):
        with self.mutex:
            frame = self.frame_data if self.frame_data is not None else None
        return frame
    
    def stop(self):
        self.started = False
        self.thread.join()
        self.socket.close()
        return self



@app.callback([Output("graph", "figure"), Output("img_1", "src"), Output("img_2", "src"), Output("img_3", "src"), Output("img_4", "src")], Input('interval-component', 'n_intervals'))
# @app.callback(Output("graph", "figure"), Input('interval-component', 'n_intervals'))
def update_bar_chart(n_intervals):
    skeletons = [interface.read_skeleton() for interface in interfaces]
    frames = [interface.read_frame() for interface in interfaces]

    kf = KalmanFilter(process_noise=0.01,
                        measurement_noise=5.0,
                        initial_estimate=0.0,)

    # kf = KalmanFilter()

    fused_skels = []
    for i in range(len(skeletons[0])):
        skel = []
        for skeleton in skeletons:
            skel.append(skeleton[i])
        fused_skels.append(kf.merge(*skel))
        # fused_skels.append(kf.step(*skel))

    # x = [pnt[0] for pnt in fused_skels]
    # y = [pnt[1] for pnt in fused_skels]
    # z = [pnt[2] for pnt in fused_skels]

    x = [pnt[0] for pnt in skeletons[0]]
    y = [pnt[1] for pnt in skeletons[0]]
    z = [pnt[2] for pnt in skeletons[0]]

    mean_x = mean([x for x in x if not np.isnan(x)])
    mean_y = mean([y for y in y if not np.isnan(y)])
    mean_z = mean([z for z in z if not np.isnan(z)])

    
    mass_center = [mean_x, mean_y, mean_z]
    scene = dict(xaxis = dict(nticks=10, range=[mass_center[0]-2, mass_center[0]+2],),
                yaxis = dict(nticks=10, range=[mass_center[1]-2, mass_center[1]+2],),
                zaxis = dict(nticks=10, range=[mass_center[2]-2, mass_center[2]+2],))

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue', size=5))])
    for (a, b) in EDGES:
        fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='lines')




    """x = [pnt[0] for pnt in skeletons[0]]
    y = [pnt[1] for pnt in skeletons[0]]
    z = [pnt[2] for pnt in skeletons[0]]
    fig.add_scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='red', size=5))
    for (a, b) in EDGES:
        fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='lines')

    x = [pnt[0] for pnt in skeletons[1]]
    y = [pnt[1] for pnt in skeletons[1]]
    z = [pnt[2] for pnt in skeletons[1]]
    fig.add_scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='green', size=5))
    for (a, b) in EDGES:
        fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='lines')"""





    fig.update_layout(showlegend=False,scene=scene, scene_camera=camera, scene_aspectmode='cube', height=1200, width=1500, margin=dict(r=20, l=20, b=10, t=10))

    ret = [fig]
    for i in range(4):
        try: 
            ret.append(frames[i])
        except:
            ret.append(None)

    return ret



# Main loop to receive data via ZeroMQ and shared_memory and update the plot
def main():
    global interfaces
    app.layout = html.Div([
                    html.H1('Skeleton tracking 3D scatter'),
                    html.Div([
                        dcc.Graph(id="graph"),
                        html.Div([
                            html.Img(id="img_1", style={"height": "300px", "width": "530px", "margin": "20 20 20 20"}),
                            html.Img(id="img_2", style={"height": "300px", "width": "530px", "margin": "20 20 20 20"}),
                            html.Img(id="img_3", style={"height": "300px", "width": "530px", "margin": "20 20 20 20"}),
                            html.Img(id="img_4", style={"height": "300px", "width": "530px", "margin": "20 20 20 20"})],
                        style={"display": "flex", "width": "100%"})],
                    style={"display": "flex", "width": "100%"}),
                    dcc.Interval(id='interval-component', interval=50, n_intervals=0)], 
                id = "change-height", 
                style={'display': 'inline-block', 'width': '100%', 'height': '100%'})
    
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(endpoint)
    s, n_devices, _ = socket.recv_string().split(" ", 2)
    socket.close()

    interfaces = [SkeletonVisualizer(n).start() for n in range(int(n_devices))]
    # webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(debug=True, port=5000)

    for interface in interfaces:
        interface.stop()
        


if __name__ == "__main__":
    main()
    