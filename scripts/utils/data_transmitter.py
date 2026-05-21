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
# Decorator
# ─────────────────────────────────────────────────────────────────────────────
def requires(mode):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.mode == mode:
                return func(self, *args, **kwargs)
            else: 
                raise AttributeError(f"'{func.__name__}' method is not enabled")
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Data transmitter
# ─────────────────────────────────────────────────────────────────────────────
class DataTransmitter:
    def __init__(self, mode: str, device_id: int, topic: str, port: int=6000):
        self.mode = mode
        self.device_id = device_id
        self.port = port+device_id
        self.topic = topic
        self.nbytes = H*W*C
        self.socket = None
        self.shm = None

        if self.mode == "sender":
            self.setup_zmq_sender()
            self.setup_shm_sender()
            self.send_frames = self._send_frames
            self.send_skeleton_data = self._send_skeleton_data
            pass
        elif self.mode == "receiver":
            pass
            self.setup_zmq_receiver()
            self.setup_shm_receiver()
            self.receive_raw_frames = self._receive_raw_frames
            self.receive_packed_skeleton_data = self._receive_packed_skeleton_data
            self.receive_frames = self._receive_frames
            self.receive_skeleton_data = self._receive_skeleton_data
        else:
            raise ValueError(f"Unknown argument: {self.mode}")
        

    # ─────────────────────────────────────────────────────────────────────────────
    # Default destructor
    # ─────────────────────────────────────────────────────────────────────────────
    def __del__(self):
        try:
            self.shutdown()
        except:
            pass
    

    # ─────────────────────────────────────────────────────────────────────────────
    # ZeroMQ setup for outgoing data
    # ─────────────────────────────────────────────────────────────────────────────
    def setup_zmq_sender(self):
        try:
            socket = zmq.Context.instance().socket(zmq.PUB)
            socket.bind(f"tcp://*:{self.port}")
            self.socket = socket
        except:
            pass


    # ─────────────────────────────────────────────────────────────────────────────
    # Shared-memory setup for outgoing data
    # ─────────────────────────────────────────────────────────────────────────────
    def setup_shm_sender(self):
        
        try:
            shm = shared_memory.SharedMemory(create=True, size=self.nbytes, name=f"shared_image{self.device_id}")
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=f"shared_image{self.device_id}")
            existing.close()
            existing.unlink()
            shm = shared_memory.SharedMemory(create=True, size=self.nbytes, name=f"shared_image{self.device_id}")
        self.shm = shm


    # Convert OpenCV image to base64 data URI
    def cv2_to_b64(self, img):
        is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not is_success: 
            return None
        encoded = base64.b64encode(buffer).decode("utf-8")
        return "data:image/jpeg;base64," + encoded
    

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
    @requires("sender")
    def _send_frames(self, frame: np.array):
        shape = frame.shape
        dtype = frame.dtype
        buf = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
        buf[:] = frame[:]


    # ─────────────────────────────────────────────────────────────────────────────
    # Send skeleton data via ZeroMQ
    # ─────────────────────────────────────────────────────────────────────────────
    @requires("sender")
    def _send_skeleton_data(self, skeleton: np.array, confidence: np.array):
        message = f"{self.topic}_{self.device_id}; {json.dumps(skeleton.tolist())}; {json.dumps(confidence.tolist())}"  # Still have to add conf
        self.socket.send_string(message)


    # ─────────────────────────────────────────────────────────────────────────────
    # Receive raw frames via shared-memory
    # ─────────────────────────────────────────────────────────────────────────────
    @requires("receiver")
    def _receive_raw_frames(self):
        frame = np.ndarray((H, W, C), dtype=np.uint8, buffer=self.shm.buf).copy()
        return frame


    # ─────────────────────────────────────────────────────────────────────────────
    # Receive raw skeleton data via ZeroMQ
    # ─────────────────────────────────────────────────────────────────────────────
    @requires("receiver")
    def _receive_packed_skeleton_data(self):
        return self.socket.recv_string()
    

    # ─────────────────────────────────────────────────────────────────────────────
    # Receive encoded frames via shared-memory
    # ─────────────────────────────────────────────────────────────────────────────
    @requires("receiver")
    def _receive_frames(self):
        # print(self.shm.name)
        frame_raw = self.receive_raw_frames()
        # print(frame_raw)
        frame = self.cv2_to_b64(frame_raw)
        return frame


    # ─────────────────────────────────────────────────────────────────────────────
    # Receive unpacked skeleton data via ZeroMQ
    # ─────────────────────────────────────────────────────────────────────────────
    @requires("receiver")
    def _receive_skeleton_data(self):
        skeleton_data_packed = self.receive_packed_skeleton_data()
        _, skeleton_packed, confidence_packed = skeleton_data_packed.split("; ", 2)
        skeleton = json.loads(skeleton_packed)
        confidence = json.loads(confidence_packed)
        return skeleton, confidence
    

    # ─────────────────────────────────────────────────────────────────────────────
    # Close shm and socket
    # ─────────────────────────────────────────────────────────────────────────────
    def shutdown(self):
        if not self.socket is None:
            self.socket.close()
        try:
            if not self.shm is None:
                rt.unregister(f"/{self.shm.name}", "shared_memory")
                self.shm.close()
                self.shm.unlink() 
        except:
            pass


        
