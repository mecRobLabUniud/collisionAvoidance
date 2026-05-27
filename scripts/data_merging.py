#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▄█░█▀▀░█▀▄░█▀▀░▀█▀░█▀█░█▀▀
░█░█░█▀█░░█░░█▀█░░░█░█░█▀▀░█▀▄░█░█░░█░░█░█░█░█
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀
"""

import time
import numpy as np
import signal
from utils.kalman_filter import KalmanFilter3D, KalmanFilter6D, ImprovedKalmanFilter6D
from utils.data_transmitter import DataTransmitter
from utils.decorators import chronometer, set_rate

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
port = 6000
running = True
topic = "SKEL"
interfaces = None
n_devices = 0
skel_len = 17
kfs = [KalmanFilter3D() for _ in range(skel_len)]


# ─────────────────────────────────────────────────────────────────────────────
# Merging
# ─────────────────────────────────────────────────────────────────────────────
# @set_rate(60)
@chronometer
def merging(dtrs, dts):
    skeletons = [dtr.receive_skeleton_data()[0] for dtr in dtrs]
    confidences = [dtr.receive_skeleton_data()[1] for dtr in dtrs]

    merged_skeleton = []
    for i in range(skel_len):
        skeleton_marker = [skeleton[i] for skeleton in skeletons if not skeleton==None]
        confidence_marker = [confidence[i] for confidence in confidences if not confidence==None]
        merged_skeleton.append(kfs[i].step(skeleton_marker, confidence_marker).tolist())

    merged_confidence = np.ones(skel_len).astype(np.float32)
    dts.send_skeleton_data(np.asanyarray(merged_skeleton), merged_confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    n_devices = 2
    dtrs = [DataTransmitter("receiver", n, "SINGLE_CAMERA") for n in range(n_devices)]
    dts = DataTransmitter("sender", n_devices, "MERGED", port=7000)
    print("Merging started correctly\n")
    
    # Main loop
    while running:
        merging(dtrs, dts)

    for dtr in dtrs:
        dtr.shutdown()
    dts.shutdown()
        

if __name__ == "__main__":
    main()