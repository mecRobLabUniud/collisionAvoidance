#!/usr/bin/env python3

"""
░█▀▀░█▀█░█▄█░█▀▀░█▀▄░█▀█░░░█▀▀░▀█▀░█▀▄░█▀▀░█▀█░█▄█
░█░░░█▀█░█░█░█▀▀░█▀▄░█▀█░░░▀▀█░░█░░█▀▄░█▀▀░█▀█░█░█
░▀▀▀░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░░░▀▀▀░░▀░░▀░▀░▀▀▀░▀░▀░▀░▀

Script for real-time skeleton tracking using RealSense D435 and YOLOv8-Pose, 
with data streaming via ZeroMQ for use in a robot control loop (e.g. admittance control). 
It includes robust depth reading, temporal smoothing of keypoints, and simplified 
capsule representation of limbs for collision avoidance. 
The script supports multiple cameras and can save video output for debugging.
"""

import signal
import os
import cv2
import numpy as np
import zmq
import time
import json
import pyrealsense2 as rs
from ultralytics import YOLO
from multiprocessing import shared_memory
from utils.skeleton_tracker import SkeletonTracker

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
W, H = 848, 480
running = True
out_port = 7000
topic = "SKEL"
save_video = False
display_stream = False
script_dir = os.path.dirname(os.path.abspath(__file__))
video_filename = os.path.join(script_dir, "../media/skeleton_tracking.avi")
yolo_model = "yolo26n-pose"


# ─────────────────────────────────────────────────────────────────────────────
# Load pose matrix
# ─────────────────────────────────────────────────────────────────────────────
def load_pose_matrix(path_txt):
    T = np.loadtxt(path_txt, dtype=np.float64)
    assert T.shape == (4, 4)
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Coordinates transformation
# ─────────────────────────────────────────────────────────────────────────────
def transform_points(T, pts_xyz):
    # pts_h: punti omogenei (N,4), aggiungendo una colonna di 1 in coda
    pts_h = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1))], axis=1)
    # T: trasformazione omogenea (4,4), pts_h.T: (4,N) (la trasposta), @ è il prodotto matriciale riga per colonna
    return (T @ pts_h.T).T[:, :3] # ritorna solo le prime 3 colonne (X,Y,Z) di tutte le righe


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton tracking
# ─────────────────────────────────────────────────────────────────────────────
def tracking(align, model, socket, video_writer):
    # Devices initialization
    ctx = rs.context()
    devices = ctx.devices  # Query connected devices
    trackers = []
    shms = []
    pose_matrixes = []
    for i, device in enumerate(devices):
        # Create trackers
        tracker = SkeletonTracker(device.get_info(rs.camera_info.serial_number)).start(align, model)
        trackers.append(tracker)

        frame = None
        while frame is None:
            frame = tracker.read_frame()
        nbytes = frame.nbytes
        shape = frame.shape
        dtype = frame.dtype

        try:
            # Create shared memory
            shm = shared_memory.SharedMemory(create=True, size=nbytes, name=f"shared_image_{i}")
        except FileExistsError:
            # If the shared memory block already exists, unlink it and create a new one
            existing_shm = shared_memory.SharedMemory(name=f"shared_image_{i}")
            existing_shm.close()
            existing_shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=nbytes, name=f"shared_image_{i}")
        shms.append(shm)

        serial = tracker.get_serial_number()
        pose_matrix = load_pose_matrix(os.path.join(script_dir, f"calibration/pose_{serial}.txt"))
        pose_matrixes.append(pose_matrix)

        print(f"Device {i} initialized: {device.get_info(rs.camera_info.name)} (SN: {device.get_info(rs.camera_info.serial_number)})")
    print("Streaming started")
    
    # Data acquisition main loop
    while running:
        for n, (tracker, shm, pose_matrix) in enumerate(zip(trackers, shms, pose_matrixes)):
            frame = tracker.read_frame()
            xyz, conf = tracker.read_coords()            

            # Write image data into shared memory
            buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            buf[:] = frame[:]

            # Trasformazione nel frame Base del Robot
            if not xyz is None and not conf is None:
                # Usa xyz_cam_s direttamente (frame ottico nativo RealSense) invece di xyz_cam_mapped
                xyz_base = transform_points(pose_matrix, xyz.astype(np.float64)).astype(np.float32)
                conf = conf.astype(np.float32)

                # if n == 1:
                #     xyz_base = np.full((17, 3), np.nan, dtype=np.float32)          

                message = f"{topic}_{n}; {len(devices)}; {json.dumps(xyz_base.tolist())}; {json.dumps(conf.tolist())}"  # Still have to add conf
                socket.send_string(message)

            if not frame is None:
                if display_stream:
                    cv2.imshow(f"YOLO Skeleton Realtime Camera {n}", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # Video saving
                if n == 0:
                    if save_video and video_writer is not None:
                        video_writer.write(frame)

    for shm in shms:
        shm.close()
        shm.unlink()  # Delete the shared memory block
    
    # Wait for both to finish
    for tracker in trackers:
        tracker.stop()

    # Resources cleanup
    print("Chiusura pipeline e finestre...")
    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
    socket.close()
    # ctx.term()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():    
    # pose_matrix = load_pose_matrix(os.path.join(script_dir, "../rotation_matrix.txt")) # Carica calibrazione camera-robot
    align = rs.align(rs.stream.color) # Allinea depth a color
    model = YOLO(os.path.join(script_dir, f"../models/{yolo_model}.engine"), verbose=False)  # Load the exported TensorRT model  

    # Inizializzazione ZeroMQ (Publisher)
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.PUB)
    socket.bind(f"tcp://*:{out_port}")

    # Inizializzazione VideoWriter
    video_writer = None
    if save_video:
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 70, (W, H))

    # Gestione segnali per chiusura pulita (es. CTRL+C o kill da script bash)
    def signal_handler(sig, frame):
        global running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    tracking(align, model, socket, video_writer)
    

if __name__ == "__main__":
    main()
