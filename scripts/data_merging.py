#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▄█░█▀▀░█▀▄░█▀▀░▀█▀░█▀█░█▀▀
░█░█░█▀█░░█░░█▀█░░░█░█░█▀▀░█▀▄░█░█░░█░░█░█░█░█
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀
"""

import sys
import zmq
import time
import numpy as np
import cv2
import os
import json
import struct
import multiprocessing.resource_tracker as rt
from multiprocessing import shared_memory
import threading
from utils.kalman_filter import KalmanFilter3D, KalmanFilter6D, ImprovedKalmanFilter6D
from utils.skeleton_receiver import SkeletonReceiver

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
r_endpoint = "tcp://localhost:6000"
s_endpoint = "tcp://*:7000"
topic = "SKEL"
interfaces = None
n_devices = 0
skel_len = 17

# Initializing kalman filter classes
# kfs = [ImprovedKalmanFilter6D() for _ in range(skel_len)]
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

    message = f"{topic}; {len(n_devices)}; {json.dumps(fused_skels)}; {json.dumps(conf)}"
    socket.send_string(message)

    print(f"Time: {time.time()-t0}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global n_devices
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(r_endpoint)
    _, n_devices, _ = socket.recv_string().split("; ", 2)
    socket.close()

    interfaces = [SkeletonReceiver(n).start() for n in range(int(n_devices))]

    # Inizializzazione ZeroMQ (Publisher)
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.PUB)
    socket.bind(s_endpoint)

    while True:
        merging(interfaces, socket)
        

if __name__ == "__main__":
    main()