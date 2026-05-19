import base64
import threading
import zmq
import mmap
import os
from flask import Flask, render_template
from flask_socketio import SocketIO
from utils.data_transmitter import DataTransmitter

TARGET_KEYPOINTS = list(range(17))  # 0..12 pelvis-up
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
dtrs = None
in_port = 7000
topic = "SKEL"

# --- ZMQ thread: receives skeleton/point data and emits to browser ---
def skeleton_thread():
    

    while True:
        try:
            merged_skeleton = dtrs[-1].receive_skeleton_data()[0]
            x = [pnt[0] for pnt in merged_skeleton]
            y = [pnt[1] for pnt in merged_skeleton]
            z = [pnt[2] for pnt in merged_skeleton]
            for (a, b) in EDGES:
                msg = {"x": [x[a], x[b]], "y": [y[a], y[b]], "z": [z[a], z[b]]}
                socketio.emit("update_scatter", msg)
        except Exception as e:
            print(f"Skeleton thread error: {e}")
        socketio.emit("display_scatter")
        socketio.sleep(0.02) 


# --- Shared-memory / shared-image thread ---
def frame_thread():
    shm_name = "/shared_image_0"          # adjust to your shm name
    shm_size = 848 *480 * 3               # adjust to your frame size

    while True:
        try:
            frames = [dtr.receive_frames() for dtr in dtrs[0:-1]]
            for n, frame in enumerate(frames):
                socketio.emit(f"update_stream{n+1}", {"frame": frame})
        except Exception as e:
            print(f"Image thread error: {e}")
        socketio.sleep(0.02)


@app.route("/")
def index():
    return render_template("index.html")


def main():
    global dtrs
    dtrs = [DataTransmitter("receiver", n, "SINGLE_CAMERA") for n in range(2)]
    dtrs.append(DataTransmitter("receiver", 2, "MERGED", port=7000))

    threading.Thread(target=skeleton_thread, daemon=True).start()              
    threading.Thread(target=frame_thread, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()