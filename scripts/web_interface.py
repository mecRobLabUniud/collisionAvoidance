#!/usr/bin/env python3

"""
░█░█░█▀▀░█▀▄░░░▀█▀░█▀█░▀█▀░█▀▀░█▀▄░█▀▀░█▀█░█▀▀░█▀▀
░█▄█░█▀▀░█▀▄░░░░█░░█░█░░█░░█▀▀░█▀▄░█▀▀░█▀█░█░░░█▀▀
░▀░▀░▀▀▀░▀▀░░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀░░░▀░▀░▀▀▀░▀▀▀

User interface for the rendering of the 3D reconstruction
of the skeleton, according to the new keypoint strcture:
0: Head 1: Left Shoulder   2: Right Shoulder  3: Left Elbow 4: Right Elbow   
5: Left Wrist  6: Right Wrist   7: Upper torso   8: Lower torso
9: Left Hip   10: Right Hip  11: Left Knee 12: Right Knee   13: Left Ankle   14: Right Ankle 
"""

import webbrowser
import threading
import numpy as np
import sys
from flask import Flask, render_template
from flask_socketio import SocketIO
from utils.data_transmitter import DataTransmitter
from utils.decorators import chronometer, set_rate

# ─────────────────────────────────────────────────────────────────────────────
# Parameters 
# ─────────────────────────────────────────────────────────────────────────────
TARGET_KEYPOINTS = list(range(15))
COCO_SKELETON = [(0, 7), (1, 3), (2, 4), (3, 5), 
                (4, 6), (7, 8), (9, 11), (10, 12), (11, 13), (12, 14)]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
dtrs = None
in_port = 7000
topic = "SKEL"


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton thread 
# ─────────────────────────────────────────────────────────────────────────────
@set_rate(30)
def send_skeleton_data():
    try:
        merged_skeleton = dtrs[-1].receive_skeleton_data()[0]
        x = [pnt[0] if not np.isnan(pnt[0]) else None for pnt in merged_skeleton]
        y = [pnt[1] if not np.isnan(pnt[1]) else None for pnt in merged_skeleton]
        z = [pnt[2] if not np.isnan(pnt[2]) else None for pnt in merged_skeleton]

        msg = {"x": x, "y": y, "z": z}
        socketio.emit("update_plot", msg)
        
    except Exception as e:
        print(f"Skeleton thread error: {e}")

def skeleton_thread():
    while True:
        send_skeleton_data()


# ─────────────────────────────────────────────────────────────────────────────
# Frame thread
# ─────────────────────────────────────────────────────────────────────────────
@set_rate(30)
def send_frames():
    try:
        frames = [dtr.receive_frames() for dtr in dtrs[0:-1]]
        for n, frame in enumerate(frames):
            socketio.emit(f"update_stream{n+1}", {"frame": frame})
    except Exception as e:
        print(f"Image thread error: {e}")

def frame_thread():
    while True:
        send_frames()
        

# ─────────────────────────────────────────────────────────────────────────────
# Web interface route
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global dtrs, n_devices
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    if arg1 is None:
        raise ValueError("No argument provided. Enter the number of cameras")   
    else:
        try:
            n_devices = int(arg1)  
        except:
            raise ValueError(f"Wrong argument: {arg1}")
        
    dtrs = [DataTransmitter("receiver", n, "SINGLE_CAMERA") for n in range(n_devices)]
    dtrs.append(DataTransmitter("receiver", n_devices, "MERGED", port=7000))

    threading.Thread(target=skeleton_thread, daemon=True).start()              
    threading.Thread(target=frame_thread, daemon=True).start()
    webbrowser.open_new('http://127.0.0.1:5000/')
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
    

if __name__ == "__main__":
    main()