#!/usr/bin/env python3


import time
import numpy as np
import signal
from utils.kalman_filter import KalmanFilter3D, KalmanFilter6D, ImprovedKalmanFilter6D
from utils.data_transmitter import DataTransmitter
from utils.decorators import chronometer, set_rate



# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    n_devices = 1
    #dtr = DataTransmitter("receiver", n_devices, "SINGLE_CAMERA") 
    dts = DataTransmitter("sender", n_devices, "SINGLE_CAMERA")

    while True:
        print('\rRunning...', end="")
        dts.send_skeleton_data(np.asanyarray([0,0]), np.asanyarray([0,0]))
        time.sleep(1)

    #dtr.shutdown()
    dts.shutdown()
        

if __name__ == "__main__":
    main()