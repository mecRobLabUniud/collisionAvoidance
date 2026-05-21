import base64
import threading
import numpy as np
import time
import os
from flask import Flask, render_template
from flask_socketio import SocketIO
from utils.data_transmitter import DataTransmitter

# ─────────────────────────────────────────────────────────────────────────────
# Parameters 
# ─────────────────────────────────────────────────────────────────────────────
TARGET_KEYPOINTS = list(range(17))  # 0..12 pelvis-up
COCO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), 
                (4, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 6), (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
dtrs = None
in_port = 7000
topic = "SKEL"


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton thread 
# ─────────────────────────────────────────────────────────────────────────────
def skeleton_thread():
    while True:
        try:
            t0 = time.time()
            merged_skeleton = dtrs[-1].receive_skeleton_data()[0]
            x = [pnt[0] for pnt in merged_skeleton]
            y = [pnt[1] for pnt in merged_skeleton]
            z = [pnt[2] for pnt in merged_skeleton]
            x_data =[]
            y_data =[]
            z_data =[]
            for (a, b) in EDGES:
                x_data.append(x[a] if not np.isnan(x[a]) else None)
                x_data.append(x[b] if not np.isnan(x[b]) else None)
                x_data.append(None)
                y_data.append(y[a] if not np.isnan(y[a]) else None)
                y_data.append(y[b] if not np.isnan(y[b]) else None)
                y_data.append(None)
                z_data.append(z[a] if not np.isnan(z[a]) else None)
                z_data.append(z[b] if not np.isnan(z[b]) else None)
                z_data.append(None)

            msg = {"x": x_data, "y": y_data, "z": z_data}
            socketio.emit("update_plot", msg)
            print(f"\rLoop time: {time.time()-t0}", end="")
        except Exception as e:
            print(f"Skeleton thread error: {e}")
        time.sleep(0.01) 


# ─────────────────────────────────────────────────────────────────────────────
# Frame thread
# ─────────────────────────────────────────────────────────────────────────────
def frame_thread():
    while True:
        try:
            frames = [dtr.receive_frames() for dtr in dtrs[0:-1]]
            for n, frame in enumerate(frames):
                socketio.emit(f"update_stream{n+1}", {"frame": frame})
        except Exception as e:
            print(f"Image thread error: {e}")
        time.sleep(0.01)


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
    global dtrs
    n_devices = 2
    dtrs = [DataTransmitter("receiver", n, "SINGLE_CAMERA") for n in range(n_devices)]
    dtrs.append(DataTransmitter("receiver", n_devices, "MERGED", port=7000))

    threading.Thread(target=skeleton_thread, daemon=True).start()              
    threading.Thread(target=frame_thread, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()