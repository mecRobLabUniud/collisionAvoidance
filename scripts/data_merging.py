#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▄█░█▀▀░█▀▄░█▀▀░▀█▀░█▀█░█▀▀
░█░█░█▀█░░█░░█▀█░░░█░█░█▀▀░█▀▄░█░█░░█░░█░█░█░█
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀
"""

import zmq
import time
import json
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
def merging(dtrs, dtss):
    t0 = time.time()

    skeletons = [dtr.receive_skeleton_data()[0] for dtr in dtrs]
    confidences = [dtr.receive_skeleton_data()[1] for dtr in dtrs]

    fused_skels = []
    for i in range(skel_len):
        skel = [skeleton[i] for skeleton in skeletons if not skeleton==None]
        conf = [confidence[i] for confidence in confidences if not confidence==None]
        fused_skels.append(kfs[i].step(skel, conf).tolist())


    print(fused_skels)
    # payload = (skeleton, confidence)
    # message = (f"{topic}_{n}; {n_devices}; "f"{json.dumps(payload[0])}; {json.dumps(payload[1])}")
    # socket.send_string(message)
    
    # message = f"MERGE_0; {n_devices}; {json.dumps(fused_skels)}; {json.dumps(None)}"
    # socket.send_string(message)

    print(f"\rLoop time: {time.time()-t0}", end="")
    time.sleep(0.2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dtrs = [DataTransmitter("receiver", n, "SINGLE_CAMERA") for n in range(2)]
    # dtss = [DataTransmitter("sender", n, "MERGED", port=7000) for n in range(2)]
    dtss = None
    print("Merging started correctly\n")
    
        # Clear shutdown logic
    def signal_handler(sig, frame):
        global running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        merging(dtrs, dtss)

    # for dtr in dtrs:
    #     dtr.shutdown()
        

if __name__ == "__main__":
    main()