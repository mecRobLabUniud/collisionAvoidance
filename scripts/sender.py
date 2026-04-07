#!/usr/bin/env python3

from multiprocessing import shared_memory
from PIL import Image
import numpy as np
import time
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from utils.skeleton_tracker import SkeletonTracker
from utils.filters import Keypoints3DSmoother
import os
import signal 


w_camera, h_camera = 848, 480
script_dir = os.path.dirname(os.path.abspath(__file__)) # Obtain the directory where this script is located
video_filename = os.path.join(script_dir, "../media/skeleton_tracking.avi")
yolo_model = "yolo26n-pose" # "yolov8x-pose.pt"

model = YOLO(os.path.join(script_dir, f"../models/{yolo_model}.engine"), verbose=False)  # Load the exported TensorRT model

   
# Inizializzazione filtri di smoothing
smoother = Keypoints3DSmoother(num_kpts=17, min_cutoff=0.1, beta=1.0)

# Devices initialization: supporta più telecamere RealSense collegate, ognuna con la propria pipeline e allineamento
align = rs.align(rs.stream.color) # Allinea depth a color

ctx = rs.context()
devices = ctx.devices  
device = devices[0]

tracker = SkeletonTracker(device.get_info(rs.camera_info.serial_number), align, model, smoother, 1).start()

frame = tracker.read_frame()
print("start")
while frame is None:
    frame = tracker.read_frame()
    pass


cv2.imshow(f"YOLO Skeleton Realtime Camera 0", frame)



# Store shape info so receiver knows dimensions
shape = frame.shape  # (H, W, 3)
dtype = frame.dtype

# Create shared memory block
shm = shared_memory.SharedMemory(create=True, size=frame.nbytes, name="shared_image")


# Write image data into shared memory
buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
buf[:] = frame[:]

print(f"Sender: image written to shared memory '{shm.name}'")
print(f"Shape: {shape}, dtype: {dtype}, size: {frame.nbytes} bytes")
print("Press Ctrl+C to release shared memory...")

try:
    while True:
        t0 = time.time()
        # time.sleep(1)

        frame = tracker.read_frame()
        while frame is None:
            frame = tracker.read_frame()
            pass


        cv2.imshow(f"YOLO Skeleton Realtime Camera 0", frame)



        # Create shared memory block
        shm = shared_memory.SharedMemory(create=False, size=frame.nbytes, name="shared_image")


        # Write image data into shared memory
        buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        buf[:] = frame[:]

        t1 = time.time()
        print(f"\rElapsed time: {t1-t0} s", end=" ")
finally:
    shm.close()
    shm.unlink()  # Delete the shared memory block
    print("Shared memory released.")