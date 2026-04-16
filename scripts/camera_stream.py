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
import struct
import zmq
import pyrealsense2 as rs
from ultralytics import YOLO
from utils.skeleton_tracker import SkeletonTracker
from utils.filters import Keypoints3DSmoother
import time
import json
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

MAGIC = b"SKEL" 
VERSION = 1
HDR_FMT = "<4sHHQ"     # magic, version, n_caps, t_mono_ns )
# REC_FMT = "<8f"        # x1 y1 z1 x2 y2 z2 radius conf
REC_FMT = "<3f"
MAX_CAPS = 32

# Parameters
w_camera, h_camera = 848, 480
running = True
arms_radius = 0.20     # Raggio della capsula (cilindro) attorno all'osso (metri)
torso_radius = 0.3     # Raggio maggiorato per la capsula del busto (metri)
endpoint = "tcp://*:6000"
topic = "SKEL"
save_video = False      # Imposta a True per salvare il video, False altrimenti
script_dir = os.path.dirname(os.path.abspath(__file__)) # Obtain the directory where this script is located
video_filename = os.path.join(script_dir, "../media/skeleton_tracking.avi")
yolo_model = "yolo26n-pose" # "yolov8x-pose.pt"
color_imgs = [] # Per visualizzazione a schermo (debug)



# Function to load the rigid transformation matrix from a text file
def load_T_base_cam(path_txt):
    T = np.loadtxt(path_txt, dtype=np.float64)
    assert T.shape == (4, 4)
    return T



# Function to apply a rigid transformation T (4x4) to a set of 3D points pts_xyz (N,3)
def transform_points(T, pts_xyz):
    # pts_h: punti omogenei (N,4), aggiungendo una colonna di 1 in coda
    pts_h = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1))], axis=1)
    # T: trasformazione omogenea (4,4), pts_h.T: (4,N) (la trasposta), @ è il prodotto matriciale riga per colonna
    return (T @ pts_h.T).T[:, :3] # ritorna solo le prime 3 colonne (X,Y,Z) di tutte le righe



# Main loop to read from camera, process skeleton and send data via ZeroMQ
def main():    
    T_base_cam = load_T_base_cam(os.path.join(script_dir, "../rotation_matrix.txt")) # Carica calibrazione camera-robot
    # --- DEBUG: Usa una matrice identità per bypassare la calibrazione errata ---    
    # T_base_cam = np.identity(4)

    # Caricamento modello YOLOv8 per Pose Estimation
    # model = YOLO(f"{yolo_model}.pt")
    model = YOLO(os.path.join(script_dir, f"../models/{yolo_model}.engine"), verbose=False)  # Load the exported TensorRT model

    # Inizializzazione ZeroMQ (Publisher)
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.PUB)
    socket.bind(endpoint)

    # Inizializzazione VideoWriter
    video_writer = None
    if save_video:
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 70, (w_camera, h_camera))

    # Inizializzazione filtri di smoothing
    smoother = Keypoints3DSmoother(num_kpts=17, min_cutoff=0.1, beta=1.0)

    # Gestione segnali per chiusura pulita (es. CTRL+C o kill da script bash)
    def signal_handler(sig, frame):
        global running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Devices initialization: supporta più telecamere RealSense collegate, ognuna con la propria pipeline e allineamento
        align = rs.align(rs.stream.color) # Allinea depth a color

        ctx = rs.context()
        devices = ctx.devices  # Query connected devices
        trackers = []
        shms = []
        for i, device in enumerate(devices):
            # Create tracker block
            tracker = SkeletonTracker(device.get_info(rs.camera_info.serial_number)).start(align, model, smoother)
            trackers.append(tracker)

            frame = None
            while frame is None:
                frame = tracker.read_frame()
            shape = frame.shape
            dtype = frame.dtype

            # Create shared memory block
            shm = shared_memory.SharedMemory(create=True, size=frame.nbytes, name=f"shared_image_{i}")
            shms.append(shm)

            print(f"Device {i} initialized: {device.get_info(rs.camera_info.name)} (SN: {device.get_info(rs.camera_info.serial_number)})")
            
        # Data acquisition main loop
        while running:
            t0 = time.time()
            xyz_base_list = []
            for n, (tracker, shm) in enumerate(zip(trackers, shms)):
                frame = tracker.read_frame()
                xyz, conf = tracker.read_coords()

                # # Create shared memory block
                # shm = shared_memory.SharedMemory(create=False, size=frame.nbytes, name="shared_image")

                # Write image data into shared memory
                buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                buf[:] = frame[:]

                # Trasformazione nel frame Base del Robot
                if not xyz is None and not conf is None:
                    # Usa xyz_cam_s direttamente (frame ottico nativo RealSense) invece di xyz_cam_mapped
                    xyz_base = transform_points(T_base_cam, xyz.astype(np.float64)).astype(np.float32)
                    conf = conf.astype(np.float32)
                    # xyz_base_list.append((xyz_base, conf))

                    payload = (xyz_base, conf)
                    message = f"{topic}_{n} {len(devices)} {json.dumps(payload[0].tolist())}"  # Still have to add conf
                    socket.send_string(message)

                # if not frame is None:
                #     cv2.imshow(f"YOLO Skeleton Realtime Camera {n}", frame)
                #     # Salvataggio video
                #     if n == 0:
                #         if save_video and video_writer is not None:
                #             video_writer.write(frame)
                # 
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            """if not xyz_base_list == [] and not xyz_base_list[0][0] is None and not frame is None:
                payload = json.dumps(xyz_base_list[0][0].tolist()) # Converti l'array numpy in lista per JSON

                message = f"{topic} {payload}"
                socket.send_string(message)

                # Misura del tempo ciclo
                tNow = time.time()
                print(f"\rTempo del nuovo ciclo main: {tNow - t0:.3f} s", end="")"""

            # # --- MODIFICA: Capsule semplificate (Braccia + Busto/Testa unico) ---
            # caps = []
            # # Helper per validità
            # def is_valid_kpt(k):
            #     return (conf[k] >= tracker.conf_thr) and np.all(np.isfinite(xyz_base[k]))
# 
            # # 1. Braccia: Spalla-Gomito (5-7, 6-8) e Gomito-Polso (7-9, 8-10)
            # arm_pairs = [(5, 7), (7, 9), (6, 8), (8, 10)]
            # for (u, v) in arm_pairs:
            #     if is_valid_kpt(u) and is_valid_kpt(v):
            #         if len(caps) >= MAX_CAPS: break
            #         pa, pb = xyz_base[u], xyz_base[v]
            #         caps.append((pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], float(arms_radius), float(min(conf[u], conf[v]))))
# 
            # # 2. Busto + Testa: Unica capsula dal punto medio dei fianchi (11,12) al naso (0)
            # if is_valid_kpt(11) and is_valid_kpt(12) and is_valid_kpt(0):
            #     if len(caps) < MAX_CAPS:
            #         p_hips = (xyz_base[11] + xyz_base[12]) * 0.5
            #         p_nose = xyz_base[0]
            #         c_torso = min(conf[11], conf[12], conf[0])
            #         caps.append((p_hips[0], p_hips[1], p_hips[2], p_nose[0], p_nose[1], p_nose[2], float(torso_radius), float(c_torso)))

            # # 5. Serializzazione e invio dati (impacchettamento e invio nel loop)
            # t_mono_ns = time.monotonic_ns()
            # # Header: Magic, Versione, Numero Capsule, Timestamp
            # # struct.pack(): converte i dati in una stringa di byte secondo il formato specificato
            # header = struct.pack(HDR_FMT, MAGIC, VERSION, len(caps), t_mono_ns)
            # # Payload: Lista di capsule (*rec serve a spacchettare la tupla della capsula in singoli argomenti - grazie all'asterisco -)
            # payload = b"".join(struct.pack(REC_FMT, *rec) for rec in caps)
            # # Invio messaggio completo (header + payload) (singolo messaggio atomico)
            # pub.send(header + payload)

        shm.close()
        shm.unlink()  # Delete the shared memory block


    finally:
        # Wait for both to finish
        for tracker in trackers:
            tracker.stop()

        # Cleanup risorse (fondamentale per non bloccare la RealSense al riavvio)
        print("Chiusura pipeline e finestre...")
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        socket.close()
        # ctx.term()



if __name__ == "__main__":
    time.sleep(2)
    main()
