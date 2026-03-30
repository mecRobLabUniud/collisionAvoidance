#!/usr/bin/env python3
 
"""
░█▀▀░█░█░█▀▀░█░░░█▀▀░▀█▀░█▀█░█▀█░░░▀█▀░█▀▄░█▀█░█▀▀░█░█░█▀▀░█▀▄
░▀▀█░█▀▄░█▀▀░█░░░█▀▀░░█░░█░█░█░█░░░░█░░█▀▄░█▀█░█░░░█▀▄░█▀▀░█▀▄
░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀░░░░▀░░▀░▀░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░▀

Script for real-time skeleton tracking using RealSense D435 and YOLOv8-Pose, 
with data streaming via ZeroMQ for use in a robot control loop (e.g. admittance control). 
It includes robust depth reading, temporal smoothing of keypoints, and simplified 
capsule representation of limbs for collision avoidance.
"""

import time
import cv2
import math
import numpy as np
import pyrealsense2 as rs
import threading
import logging

logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('tensorrt').setLevel(logging.ERROR)

TARGET_KEYPOINTS = list(range(13))  # 0..12 pelvis-up
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]

# Parameters
w_camera, h_camera = 848, 480
camera_rate = 60
conf_thr = 0.5          # Threshold of confidence for keypoint acceptance (0.0-1.0)
max_depth_range = 3.0   # Maximum depth range to consider for keypoint validation (meters)
running = True



# Function for RealSense pipeline initialization 
def camera_streaming(serial):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, w_camera, h_camera, rs.format.z16, camera_rate)
    cfg.enable_stream(rs.stream.color, w_camera, h_camera, rs.format.bgr8, camera_rate)
    pipe.start(cfg)
    return pipe



# Function to robustly read depth around a pixel (u,v) using a median filter in a neighborhood, with max distance thresholding
def robust_depth_median(depth_frame, u, v, R=6, max_dist=3.0):
    w, h = depth_frame.get_width(), depth_frame.get_height()
    uu, vv = int(round(u)), int(round(v)) # pixel centrali, round() arrotonda al più vicino intero
    zs = []
    # Aumentato R da 4 a 6 per avere più campioni su cui fare la mediana
    for dy in range(-R, R + 1):
        y = vv + dy
        if y < 0 or y >= h:
            continue
        for dx in range(-R, R + 1):
            x = uu + dx
            if x < 0 or x >= w:
                continue
            z = depth_frame.get_distance(x, y)  # metri (classe.metodo() di pyrealsense2)
            # Filtra valori zero (invalidi) e valori troppo lontani (rumorosi)
            if z > 0.05 and z <= max_dist and math.isfinite(z):
                zs.append(z)
    if not zs:
        return float("nan")
    zs.sort()
    return zs[len(zs) // 2]



# Class for tracking skeletons from RealSense camera, applying YOLOv8-Pose for keypoint detection, and using Keypoints3DSmoother for temporal smoothing and occlusion handling
class SkeletonTracker:
    def __init__(self, device, align, model, smoother, n):
        self.device = device
        self.pipe = camera_streaming(self.device)
        self.frame = None
        self.started = False
        self.xyz = None
        self.conf_thr = conf_thr
        self.conf = None
        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self.skeleton_tracking, args=(align, model, smoother, n))

    def start(self):
        if self.started:
            return
        self.started = True
        self.thread.start()
        return self

    def skeleton_tracking(self, align, model, smoother, n):
        global running
        while running and self.started:
            t0 = time.time()
            
            # Acquisizione frame (fs)
            fs = self.pipe.wait_for_frames()
            fs = align.process(fs) # Allineamento fondamentale per far corrispondere pixel RGB a Depth
            depth = fs.get_depth_frame()
            color = fs.get_color_frame()
            w_img, h_img = depth.get_width(), depth.get_height()

            if not depth or not color:
                continue

            # Conversione immagine per YOLO
            color_img = np.asanyarray(color.get_data()) # trasforma i dati grezzi della telecamera in un array NumPy
            
            # Inferenza rete neurale
            results = model.predict(color_img, verbose=False)
            
            # Se è stata rilevata almeno una persona
            if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                person = results[0].keypoints.data[0].cpu().numpy()  # (17,3) -> x, y, conf
                xy = person[:, :2]
                conf = person[:, 2]

                # Intrinseci della camera per la deproiezione
                intr = depth.profile.as_video_stream_profile().intrinsics

                # Estrazione coordinate 3D nel frame Camera
                xyz_cam = np.full((17, 3), np.nan, dtype=np.float32)
                for k in TARGET_KEYPOINTS:
                    if conf[k] < conf_thr:
                        continue
                    u, v = float(xy[k, 0]), float(xy[k, 1])
                    
                    # MODIFICA: Scarta keypoint troppo vicini ai bordi dell'immagine (lente distorta / parzialmente fuori)
                    margin = 15
                    if u < margin or u > w_img - margin or v < margin or v > h_img - margin:
                        continue

                    # Lettura robusta della profondità (con max_dist e R aumentato)
                    z = robust_depth_median(depth, u, v, R=6, max_dist=max_depth_range)
                    if not math.isfinite(z):
                        continue
                    # Deproiezione: da pixel 2D + depth, a punto 3D (metri)
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
                    xyz_cam[k] = np.array([X, Y, Z], dtype=np.float32)

                # Filtraggio temporale (OneEuroFilter)
                with self.mutex:
                    self.xyz = smoother.update(xyz_cam, conf, conf_thr)
                    self.conf = conf
                    
                # --- VISUALIZZAZIONE REAL-TIME ---
                # Disegna lo scheletro direttamente sull'immagine RGB per il debug a video
                for (u, v) in EDGES:
                    if conf[u] >= conf_thr and conf[v] >= conf_thr:
                        pt1 = (int(xy[u, 0]), int(xy[u, 1]))
                        pt2 = (int(xy[v, 0]), int(xy[v, 1]))
                        cv2.line(color_img, pt1, pt2, (0, 255, 0), 2)
                for k in TARGET_KEYPOINTS:
                    if conf[k] >= conf_thr:
                        cv2.circle(color_img, (int(xy[k, 0]), int(xy[k, 1])), 4, (0, 0, 255), -1)

            # # Misura del tempo ciclo
            # tNow = time.time()
            # if n == 1:
            #     print(f"\rTempo ciclo thread {n}: {tNow - t0:.3f} s", end="")
            # else:
            #     print(f" - Tempo ciclo thread {n}: {tNow - t0:.3f} s", end="")

            #print( f"color_img: {color_img}")
            with self.mutex:
                self.frame = color_img

    def read_frame(self):
        with self.mutex:
            frame = self.frame.copy() if self.frame is not None else None
        return frame
    
    def read_coords(self):
        with self.mutex:
            xyz = self.xyz.copy() if self.xyz is not None else None
            conf = self.conf.copy() if self.conf is not None else None
        return xyz, conf

    def stop(self):
        self.started = False
        self.thread.join()
        # self.pipe.release()