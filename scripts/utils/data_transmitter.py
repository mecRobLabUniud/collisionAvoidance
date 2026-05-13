#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░▀█▀░█▀▄░█▀█░█▀█░█▀▀░█▄█░▀█▀░▀█▀░▀█▀░█▀▀░█▀▄
░█░█░█▀█░░█░░█▀█░░░░█░░█▀▄░█▀█░█░█░▀▀█░█░█░░█░░░█░░░█░░█▀▀░█▀▄
░▀▀░░▀░▀░░▀░░▀░▀░░░░▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░░▀░░▀▀▀░▀░▀
"""

import cv2
import zmq
import json
import threading
import time
import base64
import numpy as np
import multiprocessing.resource_tracker as rt
from multiprocessing import shared_memory

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
# topic = "SKEL"
pic = None
H, W, C = 480, 848, 3 
dtype = np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# Data transmitter
# ─────────────────────────────────────────────────────────────────────────────
class DataTransmitter:
    def __init__(self, mode: str, device_id: int, port: int, topic: str):
        self.mode = mode
        self.device_id = device_id
        self.port = port
        self.topic = topic
        self.socket = None
        self.shm = None

        if self.mode == "sender":
            self.setup_zmq_sender()
            self.setup_shm_sender()
        elif self.mode == "receiver":
            self.setup_zmq_receiver()
            self.setup_shm_receiver()
        else:
            raise ValueError(f"Unknown argument: {self.mode}")
    

    # ─────────────────────────────────────────────────────────────────────────────
    # ZeroMQ setup for outgoing data
    # ─────────────────────────────────────────────────────────────────────────────
    def setup_zmq_sender(self):
        socket = zmq.Context.instance().socket(zmq.PUB)
        socket.bind(f"tcp://*:{self.port}")
        self.socket = socket


    # ─────────────────────────────────────────────────────────────────────────────
    # Shared-memory setup for outgoing data
    # ─────────────────────────────────────────────────────────────────────────────
    def setup_shm_sender(self):
        nbytes = 1221120
        try:
            shm = shared_memory.SharedMemory(create=True, size=nbytes, name=f"shared_image{self.device_id}")
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=f"shared_image{self.device_id}")
            existing.close()
            existing.unlink()
            shm = shared_memory.SharedMemory(create=True, size=nbytes, name=f"shared_image{self.device_id}")
        self.shm = shm


    # ─────────────────────────────────────────────────────────────────────────────
    # ZeroMQ setup for incoming data
    # ─────────────────────────────────────────────────────────────────────────────
    def setup_zmq_receiver(self):
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, f"{self.topic}_{self.device_id}")
        socket.connect(f"tcp://localhost:{self.port}")
        self.socket = socket


    # ─────────────────────────────────────────────────────────────────────────────
    # Shared-memory setup for incoming data
    # ─────────────────────────────────────────────────────────────────────────────
    def setup_shm_receiver(self):
        shm = shared_memory.SharedMemory(name=f"shared_image{self.device_id}")
        rt.unregister(f"/{shm.name}", "shared_memory")
        self.shm = shm


    # ─────────────────────────────────────────────────────────────────────────────
    # Send frames via shared-memory
    # ─────────────────────────────────────────────────────────────────────────────
    def send_frames(self, frame: np.array):
        shape = frame.shape
        dtype = frame.dtype
        buf = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        buf[:] = frame[:]


    # ─────────────────────────────────────────────────────────────────────────────
    # Send skeleton data via ZeroMQ
    # ─────────────────────────────────────────────────────────────────────────────
    def send_skeleton_data(self, skeleton: np.array, confidence: np.array):
        message = f"{self.topic}_{self.device_id}; {json.dumps(skeleton.tolist())}; {json.dumps(confidence.tolist())}"  # Still have to add conf
        self.socket.send_string(message)


    # @property
    # def n(self):
    #     return self.n
    # 
    # @property
    # def port(self):
    #     return self.port
    # 
    # @property
    # def topic(self):
    #     return self.topic
    



















    """
    def __init__(self, n: int, port: int, topic: str):
        zctx = zmq.Context.instance()
        socket = zctx.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, f"{topic}_{n}")
        socket.connect(f"tcp://localhost:{port}")
        self.socket = socket

        try:
            self.shm = shared_memory.SharedMemory(name=f"shared_image_{n}")
            self.remove_shm_from_resource_tracker(self.shm.name) 
        except FileNotFoundError:
            self.shm = None

        print("------------")
        print("n; ", n)
        print(self.shm)

        self.n_device_id = n
        self.started = False
        self.skeleton = None
        self.confidence = None
        self.frame = None

    def start(self):
        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self.data_receiver, args=(self.n_device_id,))
        if self.started:
            return
        self.started = True
        self.thread.start()
        return self

    # Unregister shared_memory folder
    def remove_shm_from_resource_tracker(self, name):
        rt.unregister(f"/{name}", "shared_memory")

    # Convert OpenCV image to base64 data URI
    def cv2_to_b64(self, img):
        is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not is_success: 
            return None
        encoded = base64.b64encode(buffer).decode("utf-8")
        return "data:image/jpeg;base64," + encoded

    def data_receiver(self, n: int):
        while True:
            x, _, msg1, msg2 = self.socket.recv_string().split("; ", 3)
            skeleton = json.loads(msg1)
            confidence = json.loads(msg2)
            with self.mutex:
                self.skeleton = skeleton
                self.confidence = confidence

            if not self.shm is None:
                arr = np.ndarray((H, W, C), dtype=dtype, buffer=self.shm.buf)
                img = arr.copy()
                frame = self.cv2_to_b64(img)
                with self.mutex:
                    self.frame = frame

    def read_skeleton(self):
        with self.mutex:
            skeleton = self.skeleton.copy() if self.skeleton is not None else None
        return skeleton
    
    def read_confidence(self):
        with self.mutex:
            confidence = self.confidence.copy() if self.confidence is not None else None
        return confidence
    
    def read_frame(self):
        with self.mutex:
            frame = self.frame if self.frame is not None else None
        return frame
    
    def stop(self):
        self.started = False
        self.thread.join()
        self.shm.close() 
        self.socket.close()
        return self
        """