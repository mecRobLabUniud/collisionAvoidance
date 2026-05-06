#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▀▄░█▀▀░█▀▀░█▀█░█▀▄░█▀▄░█▀▀░█▀▄
░█░█░█▀█░░█░░█▀█░░░█▀▄░█▀▀░█░░░█░█░█▀▄░█░█░█▀▀░█▀▄
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀▀░░▀▀▀░▀░▀

0: Nose
1: Left Eye
2: Right Eye
3: Left Ear
4: Right Ear
5: Left Shoulder
6: Right Shoulder
7: Left Elbow
8: Right Elbow
9: Left Wrist
10: Right Wrist
11: Left Hip
12: Right Hip
13: Left Knee
14: Right Knee
15: Left Ankle
16: Right Ankle 
"""

import sys
import zmq
import time
import numpy as np
import os
import json
import multiprocessing.resource_tracker as rt
from multiprocessing import shared_memory


TARGET_KEYPOINTS = list(range(17))  # 0..12 pelvis-up
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]

# Parameters
r_endpoint = "tcp://localhost:6000"
s_endpoint = "tcp://*:6000"
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
marker_sz = 8
line_wdt = 5
t0 = time.time()
script_dir = os.path.dirname(os.path.abspath(__file__)) # Obtain the directory where this script is located
stream_cnt = 0
n_devices = 0


def record_data(sockets, shms, skeletons_filename, frames_filename):
    for n in range(int(n_devices)):
        skeleton = sockets[n].recv_string()
        # frame = str(np.ndarray((H, W, C), dtype=np.uint8, buffer=shms[n].buf))

        skeletons_filename[n].write(skeleton + "\n")
        # frames_filename[n].write(frame + "\n")


def stream_data(socket, shms, skeletons_filename, frames_filename):
    global stream_cnt
    stream_cnt += 1
    print(stream_cnt)
    for n in range(n_devices):
        skeleton = skeletons_filename[n].readline()
        x, _, msg1, msg2 = skeleton.split("; ", 3)
        skeleton = json.loads(msg1)
        confidence = json.loads(msg2)
        #frame = frames_filename[n].readline()
        print("============================")
        print("skeleton; ", skeleton)

        # skeleton = np.asanyarray(json.loads(skeleton)) if skeleton else None
        # frame = np.asanyarray(json.loads(frame)) if frame else None     

        # Write image data into shared memory
        # buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shms[n].buf)
        # buf[:] = frame[:]

        payload = (skeleton, confidence)
        message = f"{topic}_{n}; {n_devices}; {json.dumps(payload[0])}; {json.dumps(payload[1])}"  # Still have to add conf
        socket.send_string(message)


# Main loop to receive data via ZeroMQ and shared_memory and update the plot
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if not arg is None:
        if arg in ["--record", "-r"]:
            print("Recording mode enabled. Press Ctrl+C to stop.")
            global interfaces, n_devices
            zctx = zmq.Context.instance()
            socket = zctx.socket(zmq.SUB)
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            socket.connect(r_endpoint)
            _, n_devices, _ = socket.recv_string().split("; ", 2)
            socket.close()

            sockets = []
            shms = []
            for n in range(int(n_devices)):
                zctx = zmq.Context.instance()
                socket = zctx.socket(zmq.SUB)
                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt_string(zmq.SUBSCRIBE, f"{topic}_{n}")
                socket.connect(r_endpoint)
                sockets.append(socket)

                # shm = shared_memory.SharedMemory(name=f"shared_image_{n}")
                # rt.unregister(f"/{shm.name}", "shared_memory")
                # shms.append(shm)

            # interfaces = [SkeletonReceiver(n).start() for n in range(int(n_devices))]
            skeletons_filename = [open(os.path.join(script_dir, f"data/skeleton_{n}.txt"), "w") for n in range(int(n_devices))]
            frames_filename = [open(os.path.join(script_dir, f"data/frame_{n}.txt"), "w") for n in range(int(n_devices))]
            while True:
                record_data(sockets, shms, skeletons_filename, frames_filename)
                time.sleep(0.05)






        elif arg in ["--stream", "-s"]:
            print("Streaming mode enabled. Press Ctrl+C to stop.")
            n_devices = int(len(os.listdir(os.path.join(script_dir, f"data")))/2)
            skeletons_filename = [open(os.path.join(script_dir, f"data/skeleton_{n}.txt"), "r") for n in range(int(n_devices))]
            frames_filename = [open(os.path.join(script_dir, f"data/frame_{n}.txt"), "r") for n in range(int(n_devices))]

            # Inizializzazione ZeroMQ (Publisher)
            zctx = zmq.Context.instance()
            socket = zctx.socket(zmq.PUB)
            socket.bind(s_endpoint)

            # frame = None
            # while frame is None:
            #     frame = frames_filename[0].readline()
            #     print("frame; ", json.loads(frame))
            #     frame = np.asanyarray(json.loads(frame) if frame else None)

            shms = []
            # for i in range(int(n_devices)):
            #     try:
            #         # Create shared memory block
            #         shm = shared_memory.SharedMemory(create=True, size=1, name=f"shared_image_{i}")
            #     except FileExistsError:
            #         # If the shared memory block already exists, unlink it and create a new one
            #         existing_shm = shared_memory.SharedMemory(name=f"shared_image_{i}")
            #         existing_shm.close()
            #         existing_shm.unlink()
            #         shm = shared_memory.SharedMemory(create=True, size=1, name=f"shared_image_{i}")
            #     shms.append(shm)
            while True:
                stream_data(socket, shms, skeletons_filename, frames_filename)
                time.sleep(0.05)

            for shm in shms:
                shm.close()
                shm.unlink()  # Delete the shared memory block
        else:
            raise ValueError(f"Unknown argument: {arg}")
    else:
        raise ValueError(f"No argument provided. Use --record or --stream to specify the mode.")

    quit()
        

# Entry point
if __name__ == "__main__":
    main()
    