#!/usr/bin/env python3

"""
░█░█░█▀▀░█▀▄░░░▀█▀░█▀█░▀█▀░█▀▀░█▀▄░█▀▀░█▀█░█▀▀░█▀▀
░█▄█░█▀▀░█▀▄░░░░█░░█░█░░█░░█▀▀░█▀▄░█▀▀░█▀█░█░░░█▀▀
░▀░▀░▀▀▀░▀▀░░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀░░░▀░▀░▀▀▀░▀▀▀

User interface for the rendering of the 3D reconstruction
of the skeleton, according to the following keypoint convention:
0: Nose 1: Left Eye  2: Right Eye  3: Left Ear   4: Right Ear
5: Left Shoulder   6: Right Shoulder  7: Left Elbow 8: Right Elbow   
9: Left Wrist  10: Right Wrist   11: Left Hip   12: Right Hip  
13: Left Knee 14: Right Knee   15: Left Ankle   16: Right Ankle 
"""

import plotly.graph_objs as go
import zmq
import time
import numpy as np
import webbrowser
from dash import Dash, dcc, html, Input, Output
from statistics import mean
from utils.kalman_filter import KalmanFilter3D, KalmanFilter6D, ImprovedKalmanFilter6D
from utils.skeleton_receiver import SkeletonReceiver

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
skel_len = 17
H, W, C = 480, 848, 3 
dtype = np.uint8
marker_sz = 8
line_wdt = 5
t0 = time.time()


# Launching Dash app
app = Dash(__name__)

# Initializing kalman filter classes
# kfs = [ImprovedKalmanFilter6D() for _ in range(skel_len)]
kfs = [KalmanFilter6D() for _ in range(skel_len)]


@app.callback([Output("graph", "figure"), Output("img_1", "src"), Output("img_2", "src"), Output("img_3", "src"), Output("img_4", "src")], Input('interval-component', 'n_intervals'))
# @app.callback(Output("graph", "figure"), Input('interval-component', 'n_intervals'))
def update_bar_chart(n_intervals):
    skeletons = [interface.read_skeleton() for interface in interfaces]
    confidences = [interface.read_confidence() for interface in interfaces]
    frames = [interface.read_frame() for interface in interfaces]

    fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[])])

    # print("\n================================\n")

    fused_skels = []
    for i in range(skel_len):
        skel = [skeleton[i] for skeleton in skeletons if not skeleton==None]
        conf = [confidence[i] for confidence in confidences]
        # print(f"Keypoint {i}: {[c for c in conf]}")
        fused_skels.append(kfs[i].step(skel, conf))

    x = [pnt[0] for pnt in fused_skels]
    y = [pnt[1] for pnt in fused_skels]
    z = [pnt[2] for pnt in fused_skels]
    for (a, b) in EDGES:
        fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='markers+lines', 
                          marker=dict(color='black', size=marker_sz), line=dict(color='black', width=line_wdt), opacity=0.8)
        
    # print(x, y, z)
    mean_x = mean([x for x in x if not np.isnan(x)])
    mean_y = mean([y for y in y if not np.isnan(y)])
    mean_z = mean([z for z in z if not np.isnan(z)])
    
    mass_center = [mean_x, mean_y, mean_z]
    scene = dict(xaxis = dict(nticks=10, range=[mass_center[0]-1, mass_center[0]+1],),
                yaxis = dict(nticks=10, range=[mass_center[1]-1, mass_center[1]+1],),
                zaxis = dict(nticks=10, range=[mass_center[2]-1, mass_center[2]+1],))

    """x = [pnt[0] for pnt in skeletons[0]]
    y = [pnt[1] for pnt in skeletons[0]]
    z = [pnt[2] for pnt in skeletons[0]]
    for (a, b) in EDGES:
        fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='markers+lines', 
                          marker=dict(color='grey', size=marker_sz), line=dict(color='grey', width=line_wdt), opacity=0.5)

    x = [pnt[0] for pnt in skeletons[1]]
    y = [pnt[1] for pnt in skeletons[1]]
    z = [pnt[2] for pnt in skeletons[1]]
    for (a, b) in EDGES:
        fig.add_scatter3d(x=[x[a], x[b]], y=[y[a], y[b]], z=[z[a], z[b]], mode='markers+lines', 
                          marker=dict(color='green', size=marker_sz), line=dict(color='green', width=line_wdt), opacity=0.5)"""
        
    fig.add_scatter3d(x=[0, 0.1], y=[0, 0], z=[0, 0], mode='markers+lines', 
                          marker=dict(color='red', size=marker_sz), line=dict(color='red', width=line_wdt), opacity=0.8)
    fig.add_scatter3d(x=[0, 0], y=[0, 0.1], z=[0, 0], mode='markers+lines', 
                          marker=dict(color='green', size=marker_sz), line=dict(color='green', width=line_wdt), opacity=0.8)
    fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.1], mode='markers+lines', 
                          marker=dict(color='blue', size=marker_sz), line=dict(color='blue', width=line_wdt), opacity=0.8)
    
        
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
                            html.Div([
                                html.Img(id="img_1", style={"height": "300px", "width": "530px", "margin": "20 20 20 20", "padding": "20 20 20 20"}),
                                html.Img(id="img_2", style={"height": "300px", "width": "530px", "margin": "20 20 20 20", "padding": "20 20 20 20"})],
                            style={"display": "flex", "width": "100%"}),
                            html.Div([
                                html.Img(id="img_3", style={"height": "300px", "width": "530px", "margin": "20 20 20 20", "padding": "20 20 20 20"}),
                                html.Img(id="img_4", style={"height": "300px", "width": "530px", "margin": "20 20 20 20", "padding": "20 20 20 20"})],
                            style={"display": "flex", "width": "100%"})],
                        style={"display": "inline-block", "width": "100%"})],
                    style={"display": "flex", "width": "100%"}),
                    dcc.Interval(id='interval-component', interval=100, n_intervals=0)], 
                id = "change-height", 
                style={"display": "inline-block", "width": "100%", "height": "100%"})
    
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(endpoint)
    _, n_devices, _ = socket.recv_string().split("; ", 2)
    socket.close()

    interfaces = [SkeletonReceiver(n).start() for n in range(int(n_devices))]
    # webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(debug=True, port=5000)

    for interface in interfaces:
        interface.stop()
        

# Entry point
if __name__ == "__main__":
    main()
    