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

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
port = 6000
running = True
topic = "SKEL"
interfaces = None
n_devices = 0
skel_len = 17
kfs = [KalmanFilter6D() for _ in range(skel_len)]


# ─────────────────────────────────────────────────────────────────────────────
# Merging
# ─────────────────────────────────────────────────────────────────────────────
def merging(dtrs, dts):
    t0 = time.time()

    skeletons = [dtr.receive_skeleton_data()[0] for dtr in dtrs]
    confidences = [dtr.receive_skeleton_data()[1] for dtr in dtrs]

    merged_skeleton = []
    for i in range(skel_len):
        skeleton_marker = [skeleton[i] for skeleton in skeletons if not skeleton==None]
        confidence_marker = [confidence[i] for confidence in confidences if not confidence==None]
        merged_skeleton.append(kfs[i].step(skeleton_marker, confidence_marker).tolist())

    merged_confidence = np.ones(skel_len).astype(np.float32)        
    dts.send_skeleton_data(np.asanyarray(merged_skeleton), merged_confidence)

    print(f"\rLoop time: {time.time()-t0}", end="")
    time.sleep(0.016)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dtrs = [DataTransmitter("receiver", n, "SINGLE_CAMERA") for n in range(2)]
    dts = DataTransmitter("sender", 2, "MERGED", port=7000)
    # dtss = None
    print("Merging started correctly\n")
    
        # Clear shutdown logic
    def signal_handler(sig, frame):
        global running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        merging(dtrs, dts)

    for dtr in dtrs:
        dtr.shutdown()
    dts.shutdown()
        

if __name__ == "__main__":
    main()