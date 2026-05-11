import base64
import threading
import zmq
import mmap
import os
from flask import Flask, render_template
from flask_socketio import SocketIO
from utils.skeleton_receiver import SkeletonReceiver

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
interface = None
in_port = 7000

# --- ZMQ thread: receives skeleton/point data and emits to browser ---
def zmq_thread():
    

    while True:
        skeleton = interface.read_skeleton() 
        fused_skels = skeleton[0]

        x = [pnt[0] for pnt in fused_skels]
        y = [pnt[1] for pnt in fused_skels]
        z = [pnt[2] for pnt in fused_skels]

        print( x)

        msg = {"x": [...], "y": [...], "z": [...]}
        socketio.emit("update_scatter", msg)


# --- Shared-memory / shared-image thread ---
def image_thread():
    shm_name = "/shared_image_0"          # adjust to your shm name
    shm_size = 848 *480 * 3               # adjust to your frame size

    while True:
        try:
            # fd = os.open(f"/dev/shm{shm_name}", os.O_RDONLY)
            # with mmap.mmap(fd, shm_size, access=mmap.ACCESS_READ) as shm:
            #     raw = bytes(shm[:shm_size])
            # os.close(fd)
# 
            # frame_b64 = base64.b64encode(raw).decode("utf-8")
            # socketio.emit("update_stream", {"frame": frame_b64})

            frame = interface.read_frame()
            socketio.emit("update_stream", {"frame": frame})
        except Exception as e:
            print(f"Image thread error: {e}")

        socketio.sleep(0.033)             # ~30 fps


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    

    interface = SkeletonReceiver(0, in_port).start()

    # threading.Thread(target=zmq_thread, daemon=True).start()              
    threading.Thread(target=image_thread, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)