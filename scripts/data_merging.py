#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▄█░█▀▀░█▀▄░█▀▀░▀█▀░█▀█░█▀▀
░█░█░█▀█░░█░░█▀█░░░█░█░█▀▀░█▀▄░█░█░░█░░█░█░█░█
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀
"""

import zmq
import time
import json
from utils.kalman_filter import KalmanFilter3D, KalmanFilter6D, ImprovedKalmanFilter6D
from utils.skeleton_receiver import SkeletonReceiver

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
in_port = 6000
out_port = 7000
topic = "SKEL"
interfaces = None
n_devices = 0
skel_len = 17
kfs = [KalmanFilter6D() for _ in range(skel_len)]


# ─────────────────────────────────────────────────────────────────────────────
# Merging
# ─────────────────────────────────────────────────────────────────────────────
def merging(interfaces, socket):
    t0 = time.time()
    skeletons = [interface.read_skeleton() for interface in interfaces]
    confidences = [interface.read_confidence() for interface in interfaces]

    fused_skels = []
    for i in range(skel_len):
        skel = [skeleton[i] for skeleton in skeletons if not skeleton==None]
        conf = [confidence[i] for confidence in confidences if not confidence==None]
        fused_skels.append(kfs[i].step(skel, conf).tolist())

    # payload = (skeleton, confidence)
    # message = (f"{topic}_{n}; {n_devices}; "f"{json.dumps(payload[0])}; {json.dumps(payload[1])}")
    # socket.send_string(message)
    
    message = f"{topic}_0; {n_devices}; {json.dumps(fused_skels)}; {json.dumps(None)}"
    socket.send_string(message)

    print(f"\rLoop time: {time.time()-t0}", end="")
    time.sleep(0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global n_devices
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(f"tcp://localhost:{in_port}")
    _, n_devices, _ = socket.recv_string().split("; ", 2)
    socket.close()

    interfaces = [SkeletonReceiver(n, in_port).start() for n in range(int(n_devices))]

    # Inizializzazione ZeroMQ (Publisher)
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.PUB)
    socket.bind(f"tcp://*:{out_port}")

    print("Merging started correctly\n")
    while True:
        merging(interfaces, socket)
        

if __name__ == "__main__":
    main()