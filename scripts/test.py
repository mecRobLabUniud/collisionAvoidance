#!/usr/bin/env python3

from utils.data_transmitter import DataTransmitter
import time
import cv2

out_port = 6000

while True:
    print("=========================")
    for n in range(2):
        dt = DataTransmitter("receiver", n, "SINGLE_CAMERA")
        
        color_frame = dt.receive_raw_frames()

        # color_frame is BGR, ready to use
        cv2.imshow(f"Color {n}", color_frame)

        # # depth_frame is the JET colormap — convert back to grayscale if needed
        # depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Depth", depth_gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    time.sleep(0.5)