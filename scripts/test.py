#!/usr/bin/env python3

from utils.data_transmitter import DataTransmitter
import time

out_port = 6000

while True:
    print("=========================")
    for n in range(2):
        dt = DataTransmitter("receiver", n, "SINGLE_CAMERA")
        
        print(dt.receive_packed_skeleton_data())

    time.sleep(0.5)