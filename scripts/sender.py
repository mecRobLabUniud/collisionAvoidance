#!/usr/bin/env python3

import zmq, pickle
import cv2
import numpy as np
import time


img = cv2.imread('../prova.png')  # Or capture from camera

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.bind("tcp://127.0.0.1:5555")



while True:
    sock.send(pickle.dumps(np.array(img)))
    time.sleep(1)