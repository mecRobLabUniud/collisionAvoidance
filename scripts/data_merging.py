#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▄█░█▀▀░█▀▄░█▀▀░▀█▀░█▀█░█▀▀
░█░█░█▀█░░█░░█▀█░░░█░█░█▀▀░█▀▄░█░█░░█░░█░█░█░█
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░▀▀▀
Merging and re-shaping the skeleton standard keypoint convention:
0: Nose 1: Left Eye  2: Right Eye  3: Left Ear   4: Right Ear
5: Left Shoulder   6: Right Shoulder  7: Left Elbow 8: Right Elbow   
9: Left Wrist  10: Right Wrist   11: Left Hip   12: Right Hip  
13: Left Knee 14: Right Knee   15: Left Ankle   16: Right Ankle 
"""

import sys
import numpy as np
import time
import signal
from utils.kalman_filter import SimpleMerger, KalmanFilter3D, KalmanFilter6D, ImprovedKalmanFilter6D
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
kfs = [KalmanFilter6D() for _ in range(skel_len)]

# ─────────────────────────────────────────────────────────────────────────────
# Re-shaping skeleton structure
# ─────────────────────────────────────────────────────────────────────────────
def reshape_structure(skeleton):
    new_skeleton = []
    head_markers = [wp for wp in skeleton[0:5] if not np.isnan(wp[0])]
    head = [[head_markers[i][j] for i in range(len(head_markers))] for j in range(3)]
    new_skeleton.append([sum(head[i])/len(head[i]) if len(head[i])>0 else np.nan for i in range(3)])
    for i in range(5, 11):
        new_skeleton.append(skeleton[i]) 
    upper_torso_markers = [wp for wp in skeleton[5:7] if not np.isnan(wp[0])]
    lower_torso_markers = [wp for wp in skeleton[11:13] if not np.isnan(wp[0])]
    upper_torso = [[upper_torso_markers[i][j] for i in range(len(upper_torso_markers))] for j in range(3)]
    lower_torso = [[lower_torso_markers[i][j] for i in range(len(lower_torso_markers))] for j in range(3)]
    new_skeleton.append([sum(upper_torso[i])/len(upper_torso[i]) if len(upper_torso[i])>0 else np.nan for i in range(3)])
    new_skeleton.append([sum(lower_torso[i])/len(lower_torso[i]) if len(lower_torso[i])>0 else np.nan for i in range(3)])
    for i in range(11, 17):
        new_skeleton.append(skeleton[i]) 

    new_skeleton = np.asanyarray(new_skeleton)
    return new_skeleton


# ─────────────────────────────────────────────────────────────────────────────
# Merging
# ─────────────────────────────────────────────────────────────────────────────
@set_rate(60)
def merging(dtrs, dts):
    # skeletons = [dtr.receive_skeleton_data()[0] for dtr in dtrs]
    # confidences = [dtr.receive_skeleton_data()[1] for dtr in dtrs]
    skeletons = []
    confidences = []
    for dtr in dtrs:
        skeleton, confidence = dtr.receive_skeleton_data()
        if skeleton is None or confidence is None:
            skeletons.append(None)
            confidences.append(None)
        else:
            skeletons.append(skeleton)
            confidences.append(confidence)

    merged_skeleton = []
    for i in range(skel_len):
        skeleton_marker = [skeleton[i] for skeleton in skeletons if not skeleton==None]
        confidence_marker = [confidence[i] for confidence in confidences if not confidence==None]
        merged_skeleton.append(kfs[i].step(skeleton_marker, confidence_marker).tolist())

    reshaped_skeleton = reshape_structure(merged_skeleton)    
    merged_confidence = np.ones(skel_len).astype(np.float32)
    dts.send_skeleton_data(reshaped_skeleton, merged_confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global n_devices
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    if arg1 is None:
        raise ValueError("No argument provided. Enter the number of cameras")   
    else:
        try:
            n_devices = int(arg1)  
        except:
            raise ValueError(f"Wrong argument: {arg1}")
        
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