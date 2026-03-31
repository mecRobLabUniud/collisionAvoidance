#!/usr/bin/env python3

import zmq, pickle
import cv2
import numpy as np
import time


ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.connect("tcp://127.0.0.1:5555")



while True:
    arr = pickle.loads(sock.recv())
    

    cv2.imshow("YOLO Skeleton Realtime Camera", arr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break