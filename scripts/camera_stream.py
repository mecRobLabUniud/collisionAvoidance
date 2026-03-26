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
import pyrealsense2 as rs
from ultralytics import YOLO
from utils.skeleton_tracker import SkeletonTracker
from utils.filters import Keypoints3DSmoother, Keypoints3DKalmanFilter
import logging

logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('tensorrt').setLevel(logging.ERROR)

MAGIC = b"SKEL" 
VERSION = 1
HDR_FMT = "<4sHHQ"     # magic, version, n_caps, t_mono_ns )
REC_FMT = "<8f"        # x1 y1 z1 x2 y2 z2 radius conf
MAX_CAPS = 32

# Parameters
w_camera, h_camera = 848, 480
running = True
arms_radius = 0.20     # Raggio della capsula (cilindro) attorno all'osso (metri)
torso_radius = 0.3     # Raggio maggiorato per la capsula del busto (metri)
endpoint = "ipc:///tmp/skeleton.ipc" # Indirizzo socket ZeroMQ (IPC per comunicazione locale veloce)
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
    pub = zctx.socket(zmq.PUB) # socket di tipo Publisher (trasmette dati a chiunque sia connesso, se nessuno è connesso i dati vengono persi)
    pub.setsockopt(zmq.LINGER, 0) # Evita che ZMQ blocchi la chiusura se ci sono messaggi pendenti
    pub.bind(endpoint) # Associa il socket all'endpoint specificato (questo script python crea e possiede il socket, gli altri processi si connettono a questo endpoint, come il cpp del controllo ammettenza)

    # Inizializzazione VideoWriter
    video_writer = None
    if save_video:
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 70, (w_camera, h_camera))

    # Inizializzazione filtri di smoothing
    # Option 1: Use One Euro Filter (original)
    # smoother = Keypoints3DSmoother(num_kpts=17, min_cutoff=0.1, beta=1.0)
    
    # Option 2: Use Kalman Filter (recommended for better tracking)
    smoother = Keypoints3DKalmanFilter(num_kpts=17, process_variance=0.01, measurement_variance=0.1)

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
        for i, device in enumerate(devices):
            trackers.append(SkeletonTracker(device.get_info(rs.camera_info.serial_number), align, model, smoother, i+1).start())
            print(f"Device {i} initialized: {device.get_info(rs.camera_info.name)} (SN: {device.get_info(rs.camera_info.serial_number)})")

        while running:
            for n, tracker in enumerate(trackers):
                frame = tracker.read_frame()
                xyz, conf = tracker.read_coords()

                # Trasformazione nel frame Base del Robot
                if not xyz is None and not conf is None:
                    # Usa xyz_cam_s direttamente (frame ottico nativo RealSense) invece di xyz_cam_mapped
                    xyz_base = transform_points(T_base_cam, xyz.astype(np.float64)).astype(np.float32)
                    conf = conf.astype(np.float32)

                    # --- MODIFICA: Capsule semplificate (Braccia + Busto/Testa unico) ---
                    caps = []
                    # Helper per validità
                    def is_valid_kpt(k):
                        return (conf[k] >= tracker.conf_thr) and np.all(np.isfinite(xyz_base[k]))

                    # 1. Braccia: Spalla-Gomito (5-7, 6-8) e Gomito-Polso (7-9, 8-10)
                    arm_pairs = [(5, 7), (7, 9), (6, 8), (8, 10)]
                    for (u, v) in arm_pairs:
                        if is_valid_kpt(u) and is_valid_kpt(v):
                            if len(caps) >= MAX_CAPS: break
                            pa, pb = xyz_base[u], xyz_base[v]
                            caps.append((pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], float(arms_radius), float(min(conf[u], conf[v]))))

                    # 2. Busto + Testa: Unica capsula dal punto medio dei fianchi (11,12) al naso (0)
                    if is_valid_kpt(11) and is_valid_kpt(12) and is_valid_kpt(0):
                        if len(caps) < MAX_CAPS:
                            p_hips = (xyz_base[11] + xyz_base[12]) * 0.5
                            p_nose = xyz_base[0]
                            c_torso = min(conf[11], conf[12], conf[0])
                            caps.append((p_hips[0], p_hips[1], p_hips[2], p_nose[0], p_nose[1], p_nose[2], float(torso_radius), float(c_torso)))

                    # # 5. Serializzazione e invio dati (impacchettamento e invio nel loop)
                    # t_mono_ns = time.monotonic_ns()
                    # # Header: Magic, Versione, Numero Capsule, Timestamp
                    # # struct.pack(): converte i dati in una stringa di byte secondo il formato specificato
                    # header = struct.pack(HDR_FMT, MAGIC, VERSION, len(caps), t_mono_ns)
                    # # Payload: Lista di capsule (*rec serve a spacchettare la tupla della capsula in singoli argomenti - grazie all'asterisco -)
                    # payload = b"".join(struct.pack(REC_FMT, *rec) for rec in caps)
                    # # Invio messaggio completo (header + payload) (singolo messaggio atomico)
                    # pub.send(header + payload)
                
                if not frame is None:
                    cv2.imshow(f"YOLO Skeleton Realtime Camera {n}", frame)
                    # Salvataggio video
                    if save_video and video_writer is not None:
                        video_writer.write(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Wait for both to finish
        for tracker in trackers:
            tracker.stop()

        # Cleanup risorse (fondamentale per non bloccare la RealSense al riavvio)
        print("Chiusura pipeline e finestre...")
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        pub.close()
        # ctx.term()



if __name__ == "__main__":
    main()
