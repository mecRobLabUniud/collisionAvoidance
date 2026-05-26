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

import os
import cv2
import math
import time
import numpy as np
import pyrealsense2 as rs
import threading
import logging
from utils.filters import Keypoints3DSmoother
from utils.decorators import chronometer, set_rate

logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('tensorrt').setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
TARGET_KEYPOINTS = list(range(17))  # 0..12 pelvis-up
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]
conf_thr = 0.5          # Threshold of confidence for keypoint acceptance (0.0-1.0)
max_depth_range = 3.0   # Maximum depth range to consider for keypoint validation (meters)
running = True


# color_writer = cv2.VideoWriter("color.avi", cv2.VideoWriter_fourcc(*'XVID'), 60, (848, 480))
# depth_writer = cv2.VideoWriter("depth.avi", cv2.VideoWriter_fourcc(*'XVID'), 60, (848, 480), isColor=True)


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton tracker using YOLOv8-Pose for keypoint detection
# ─────────────────────────────────────────────────────────────────────────────
class SkeletonTracker:
    def __init__(self, device, w_camera: int=848, h_camera: int=480, camera_rate: int=60, depth: bool=True, save: bool=False):
        self.device = device
        self.align = rs.align(rs.stream.color) # Allinea depth a color
        self.pipe = self.setup_camera_streaming(self.device, w_camera, h_camera, camera_rate, depth)
        self.color = None
        self.depth = None
        self.frame = None
        self.camera_thread = None
        self.model_thread = None
        self.started = False
        self.xyz = None
        self.conf_thr = conf_thr
        self.conf = None
        self.smoother = Keypoints3DSmoother(num_kpts=17, min_cutoff=0.1, beta=1.0)
        self.save_data = save
        

    # ─────────────────────────────────────────────────────────────────────────────
    # Default destructor
    # ─────────────────────────────────────────────────────────────────────────────
    def __del__(self):
        try:
            self.shutdown()
        except:
            pass


    # ─────────────────────────────────────────────────────────────────────────────
    # RealSense pipeline initialization 
    # ─────────────────────────────────────────────────────────────────────────────
    def setup_camera_streaming(self, serial, w_camera, h_camera, camera_rate, depth):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        if depth:
            cfg.enable_stream(rs.stream.depth, w_camera, h_camera, rs.format.z16, camera_rate)
        cfg.enable_stream(rs.stream.color, w_camera, h_camera, rs.format.bgr8, camera_rate)
        pipe.start(cfg)
        return pipe


    def start(self, model):
        self.mutex = threading.Lock()
        if self.started:
            return
        self.started = True
        self.camera_thread = threading.Thread(target=self.camera_streaming, args=())
        self.model_thread = threading.Thread(target=self.skeleton_tracking, args=(model,))

        self.camera_thread.start()
        self.model_thread.start()
        return self
    

    def camera_streaming(self):
        last_frame_number = -1
        while running and self.started:
            fs = self.pipe.wait_for_frames()
            fs = self.align.process(fs)
            depth = fs.get_depth_frame()
            color = fs.get_color_frame()

            if not depth or not color:
                continue

            frame_number = color.get_frame_number()
            if frame_number == last_frame_number:
                continue
            last_frame_number = frame_number

            self.W, self.H = depth.get_width(), depth.get_height()
            self.intr = depth.profile.as_video_stream_profile().intrinsics
            with self.mutex:
                self.color = color
                self.depth = depth


    def skeleton_tracking(self, model):
        while running and self.started:
            if not self.depth or not self.color:
                continue
            else:
                with self.mutex:
                    color = self.color
                    depth = self.depth

            # Neural network inference
            color_img = np.asanyarray(color.get_data())
            results = model.predict(color_img, verbose=False)

            if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                person = results[0].keypoints.data[0].cpu().numpy()  # (17,3) -> x, y, conf
                xy = person[:, :2]
                conf = person[:, 2]
                xyz_cam = np.full((17, 3), np.nan, dtype=np.float32)
                for k in TARGET_KEYPOINTS:
                    if conf[k] < conf_thr:
                        continue
                    u, v = float(xy[k, 0]), float(xy[k, 1])
                    margin = 15
                    if u < margin or u > self.W - margin or v < margin or v > self.H - margin:
                        continue

                    # Depth reading
                    z = self.robust_depth_median(depth, u, v, R=6, max_dist=max_depth_range)
                    if not math.isfinite(z):
                        continue
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(self.intr, [u, v], z)
                    xyz_cam[k] = np.array([X, Y, Z], dtype=np.float32)

                # Temporal filter
                with self.mutex:
                    self.xyz = self.smoother.update(xyz_cam, conf, conf_thr)
                    self.conf = conf
                    
                # Image annotation for debugging
                for (u, v) in EDGES:
                    if conf[u] >= conf_thr and conf[v] >= conf_thr:
                        pt1 = (int(xy[u, 0]), int(xy[u, 1]))
                        pt2 = (int(xy[v, 0]), int(xy[v, 1]))
                        cv2.line(color_img, pt1, pt2, (0, 255, 0), 2)
                for k in TARGET_KEYPOINTS:
                    if conf[k] >= conf_thr:
                        cv2.circle(color_img, (int(xy[k, 0]), int(xy[k, 1])), 4, (0, 0, 255), -1)

            with self.mutex:
                self.frame = color_img.copy()


    # ─────────────────────────────────────────────────────────────────────────────
    # Robust depth reading around a pixel 
    # ─────────────────────────────────────────────────────────────────────────────
    def robust_depth_median(self, depth_frame, u, v, R=6, max_dist=3.0):
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


    def get_aligned_frames(self):
        depth = None
        color = None
        while depth is None and color is None:
            fs = self.pipe.wait_for_frames()
            fs = self.align.process(fs)
            depth = fs.get_depth_frame()
            color = fs.get_color_frame()

        depth = np.asanyarray(depth.get_data()) 
        color = np.asanyarray(color.get_data()) 
        return depth, color
    

    def get_depth_frame(self):
        depth = None
        while depth is None:
            fs = self.pipe.wait_for_frames()
            depth = fs.get_depth_frame()

        depth = np.asanyarray(depth.get_data()) 
        return depth
    

    def get_color_frame(self):
        color = None
        while color is None:
            fs = self.pipe.wait_for_frames()
            color = fs.get_color_frame()

        color = np.asanyarray(color.get_data()) 
        return color
    

    def get_serial_number(self):
        return self.device
    

    def get_intrinsics(self):
        color_stream = self.pipe.get_active_profile().get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        mtx = np.array([
            [intrinsics.fx, 0,             intrinsics.ppx],
            [0,              intrinsics.fy, intrinsics.ppy],
            [0,              0,             1]
        ], dtype=np.float64)
        dist = np.array(intrinsics.coeffs, dtype=np.float64) 
        return mtx, dist


    def read_frame(self):
        with self.mutex:
            frame = self.frame.copy() if self.frame is not None else None
        return frame
    

    def read_coords(self):
        with self.mutex:
            xyz = self.xyz.copy() if self.xyz is not None else None
            conf = self.conf.copy() if self.conf is not None else None            
        return xyz, conf


    def shutdown(self):
        self.started = False
        self.camera_thread.join()
        self.model_thread.join()
        if self.save_data:            
            self.color_writer.release()
            self.depth_writer.release()
            self.frame_writer.release()

        

        