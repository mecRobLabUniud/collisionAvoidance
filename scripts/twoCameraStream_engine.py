#!/usr/bin/env python3
"""
Script per il rilevamento dello skeleton umano e trasmissione dati via ZeroMQ.
Utilizza YOLOv8 per il pose estimation da camera RealSense, applica filtri One Euro
per ridurre il jitter, converte i keypoints in capsule geometriche e trasmette
i dati al processo C++ per il controllo di sicurezza del robot.
"""

import time
import signal
import os
import cv2
import math
import struct
import numpy as np
import zmq
import pyrealsense2 as rs
from ultralytics import YOLO
import threading


""" Tips to speed up code
Better alternatives to try first:
Profile to find real hotspots (cProfile, timeit) — I'll do this if you want.
Use Numba to JIT-compile Python loops (great for small arrays and per-frame loops).
Use CuPy for a near-drop-in GPU-accelerated NumPy replacement if you have a CUDA GPU and can batch operations.
Vectorize more operations (reduce Python-level loops) so existing NumPy can exploit BLAS. """




mutex = threading.Lock()
running = True

MAGIC = b"SKEL" 
VERSION = 1
HDR_FMT = "<4sHHQ"     # magic, version, n_caps, t_mono_ns )
REC_FMT = "<8f"        # x1 y1 z1 x2 y2 z2 radius conf
MAX_CAPS = 32

# ---------- Config (coerente coi tuoi script) ----------
# Definizione dei keypoint di interesse (es. escludendo piedi se non servono)
TARGET_KEYPOINTS = list(range(13))  # 0..12 pelvis-up (fino alle anche)
# Definizione delle connessioni (ossa) basata sullo standard COCO
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
# Filtra solo le connessioni che coinvolgono i keypoint target
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]

# Parametri rapidi
conf_thr = 0.5          # Soglia di confidenza minima per considerare valido un keypoint
arms_radius = 0.20     # Raggio della capsula (cilindro) attorno all'osso (metri)
torso_radius = 0.3     # Raggio maggiorato per la capsula del busto (metri)
max_depth_range = 3.0   # [m] Ignora punti oltre questa distanza (D435 diventa rumorosa)
endpoint = "ipc:///tmp/skeleton.ipc" # Indirizzo socket ZeroMQ (IPC per comunicazione locale veloce)
save_video = False      # Imposta a True per salvare il video, False altrimenti
script_dir = os.path.dirname(os.path.abspath(__file__)) # Obtain the directory where this script is located
video_filename = os.path.join(script_dir, "../media/outputSkeletonTracking.avi")
wCamera, hCamera = 848, 480
cameraRate = 60
yoloModel = "yolo26n-pose" # "yolov8x-pose.pt"
color_imgs = [] # Per visualizzazione a schermo (debug)
results = []
colors = []
depths = []



# ---------- Filtro One Euro ----------
# Implementazione del filtro 1€ (One Euro Filter) per smoothing adattivo dei keypoints.
# Riduce il jitter nei movimenti lenti mantenendo bassa latenza nei movimenti veloci.
class OneEuroFilter:
    # Costruttore del filtro
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)  # Cutoff minimo per segnali quasi fermi
        self.beta = float(beta)              # Coefficiente di adattamento alla velocità
        self.d_cutoff = float(d_cutoff)      # Cutoff per la derivata (velocità)
        self.x_prev = float(x0)              # Ultimo valore filtrato
        self.dx_prev = float(dx0)            # Ultima velocità stimata
        self.t_prev = float(t0)              # Timestamp ultimo aggiornamento

    # Calcola il fattore di smoothing basato su tempo e cutoff
    def smoothing_factor(self, t_e, cutoff):
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    # Applica smoothing esponenziale
    def exponential_smoothing(self, alpha, x, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    # Aggiorna il filtro con nuovo campione (t, x)
    def __call__(self, t, x):
        t_e = t - self.t_prev  # Tempo trascorso (dt)
        if t_e <= 0.0:
            return self.x_prev
        # Stima velocità con smoothing
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        # Adatta cutoff basato sulla velocità
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        # Filtra posizione
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        # Aggiorna stato
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat



# Classe per gestire il filtraggio 3D indipendente dei keypoints dello skeleton.
# Crea filtri One Euro separati per ogni coordinata (x,y,z) di ogni keypoint.
# Gestisce occlusioni mantenendo gli ultimi valori validi per breve tempo.
class Keypoints3DSmoother:
    # Costruttore: inizializza parametri e strutture dati
    def __init__(self, num_kpts=17, min_cutoff=0.1, beta=1.0):
        self.num_kpts = num_kpts  # Numero di keypoints (default 17 per COCO pose)
        self.min_cutoff = min_cutoff  # Cutoff minimo per filtri
        self.beta = beta              # Beta per adattamento velocità
        self.t0 = time.monotonic()    # Tempo di riferimento iniziale
        self.initialized = False      # Flag per inizializzazione filtri
        self.filters = []             # Lista di tuple (filter_x, filter_y, filter_z) per keypoint
        self.last_valid = np.full((num_kpts, 3), np.nan, dtype=np.float32)  # Ultimi valori validi
        self.last_valid_time = np.zeros(num_kpts, dtype=np.float64)         # Timestamp ultimi valori validi

    # Metodo di aggiornamento: applica filtri ai nuovi keypoints
    def update(self, xyz, conf, conf_thr):
        t = time.monotonic() - self.t0  # Tempo relativo
        # Inizializzazione lazy dei filtri al primo frame valido
        if not self.initialized:
            for i in range(self.num_kpts):
                x0 = float(xyz[i, 0]) if np.isfinite(xyz[i, 0]) else 0.0
                y0 = float(xyz[i, 1]) if np.isfinite(xyz[i, 1]) else 0.0
                z0 = float(xyz[i, 2]) if np.isfinite(xyz[i, 2]) else 0.0
                self.filters.append((
                    OneEuroFilter(t, x0, min_cutoff=self.min_cutoff, beta=self.beta),
                    OneEuroFilter(t, y0, min_cutoff=self.min_cutoff, beta=self.beta),
                    OneEuroFilter(t, z0, min_cutoff=self.min_cutoff, beta=self.beta),
                ))
            self.initialized = True

        out = np.copy(xyz).astype(np.float32)  # Copia per output
        for i in range(self.num_kpts):
            # Controlla validità del keypoint (confidenza e finitezza)
            valid = (conf[i] >= conf_thr) and np.all(np.isfinite(xyz[i]))
            
            if not valid:
                # Gestione occlusioni: usa ultimi valori validi se recenti (max 0.5s)
                # Se il dato manca per più di 0.5s, smettiamo di predire e restituiamo NaN.
                if np.all(np.isfinite(self.last_valid[i])) and (t - self.last_valid_time[i] < 0.5):
                    out[i] = self.last_valid[i]
                else:
                    out[i] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                continue
            # Applica il filtro su ogni asse
            fx, fy, fz = self.filters[i] # nota che fx,fy,fz sono istanze (oggetti) di OneEuroFilter!
            out[i, 0] = fx(t, float(xyz[i, 0]))
            out[i, 1] = fy(t, float(xyz[i, 1]))
            out[i, 2] = fz(t, float(xyz[i, 2]))
            self.last_valid[i] = out[i]
            self.last_valid_time[i] = t
        return out



# Estrae la profondità (Z) in modo robusto calcolando la mediana dei valori di profondità
# in una finestra RxR attorno al pixel (u, v).
# Questo aiuta a ignorare i pixel con profondità mancante (0) o rumorosa.
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



# Carica la matrice di trasformazione omogenea (4x4) dal file TXT.
def load_T_base_cam(path_txt):
    T = np.loadtxt(path_txt, dtype=np.float64)
    assert T.shape == (4, 4)
    return T



# Applica una trasformazione rigida (rotazione + traslazione) ai punti 3D.
# Usata per passare dal sistema di riferimento della telecamera a quello della base del robot.
def transform_points(T, pts_xyz):
    # pts_h: punti omogenei (N,4), aggiungendo una colonna di 1 in coda
    pts_h = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1))], axis=1)
    # T: trasformazione omogenea (4,4), pts_h.T: (4,N) (la trasposta), @ è il prodotto matriciale riga per colonna
    return (T @ pts_h.T).T[:, :3] # ritorna solo le prime 3 colonne (X,Y,Z) di tutte le righe

# ---------- ZMQ message: header + records ----------
# Definizione del protocollo binario per la trasmissione dati
# MAGIC = b"SKEL" # Identificatore del messaggio (4 byte) (il pacchetto inizia con questi 4 byte)
# HDR_FMT = "<4sHHQ": Definisce la struttura dell'Intestazione (Header) del pacchetto.
# <: Indica Little-Endian (ordine byte)
# 4s: 4 byte stringa (MAGIC) (per SKEL)
# H: unsigned short (2 byte) (VERSION)
# H: unsigned short (2 byte) (n_caps, numero di capsule nel messaggio)
# Q: unsigned long long (8 byte) (t_mono_ns, timestamp in nanosecondi)
# REC_FMT = "<8f": Definisce la struttura di ogni record di capsula.
# <: Little-Endian
# 8f: 8 float (4 byte ciascuno) (x1, y1, z1 per l'inizio, x2, y2, z2 per la fine, radius, conf)



def cameraStreaming(serial):
    align = rs.align(rs.stream.color) # Allinea depth a color

    # Inizializzazione pipeline RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, wCamera, hCamera, rs.format.z16, cameraRate)
    cfg.enable_stream(rs.stream.color, wCamera, hCamera, rs.format.bgr8, cameraRate)
    pipe.start(cfg)

    return pipe



def skeletonTracking(pipe, align, model, n):
    global color_imgs, running
    while running:
        t0 = time.time()
        
        # Acquisizione frame (fs)
        fs = pipe.wait_for_frames()
        fs = align.process(fs) # Allineamento fondamentale per far corrispondere pixel RGB a Depth
        depth = fs.get_depth_frame()
        color = fs.get_color_frame()

        if not depth or not color:
            continue

        # Conversione immagine per YOLO
        color_img = np.asanyarray(color.get_data()) # trasforma i dati grezzi della telecamera in un array NumPy
        
        # Inferenza rete neurale
        # with mutex:
        result = model.predict(color_img, verbose=False)
        
        color_imgs[n-1] = color_img # Salva l'immagine processata per la visualizzazione nel main
        results[n-1] = result
        colors[n-1] = color
        depths[n-1] = depth



def main():    
    T_base_cam = load_T_base_cam(os.path.join(script_dir, "../rotation_matrix.txt")) # Carica calibrazione camera-robot
    # --- DEBUG: Usa una matrice identità per bypassare la calibrazione errata ---    
    # T_base_cam = np.identity(4)

    # Caricamento modello YOLOv8 per Pose Estimation
    model = YOLO(f"{yoloModel}.pt")
    model.export(format="engine")  # Export the model to TensorRT format
    tensorRT_model = YOLO(f"{yoloModel}.engine")  # Load the exported TensorRT model

    # Inizializzazione ZeroMQ (Publisher)
    zctx = zmq.Context.instance()
    pub = zctx.socket(zmq.PUB) # socket di tipo Publisher (trasmette dati a chiunque sia connesso, se nessuno è connesso i dati vengono persi)
    pub.setsockopt(zmq.LINGER, 0) # Evita che ZMQ blocchi la chiusura se ci sono messaggi pendenti
    pub.bind(endpoint) # Associa il socket all'endpoint specificato (questo script python crea e possiede il socket, gli altri processi si connettono a questo endpoint, come il cpp del controllo ammettenza)

    # Inizializzazione VideoWriter
    video_writer = None
    if save_video:
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 70, (wCamera, hCamera))

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
        pipes = []
        for i, device in enumerate(devices):
            pipes.append(cameraStreaming(device.get_info(rs.camera_info.serial_number)))
            print(f"Device {i} initialized: {device.get_info(rs.camera_info.name)} (SN: {device.get_info(rs.camera_info.serial_number)})")
        
        # Create and start threads
        threads = []
        for n, pipe in enumerate(pipes):
            thread = threading.Thread(target=skeletonTracking, args=(pipe, align, tensorRT_model, n))
            color_imgs.append(None) # Inizializza la lista delle immagini per la visualizzazione
            results.append(None)
            colors.append(None)
            depths.append(None)
            thread.start()
            threads.append(thread)

            

        while running:
            for n, (color_img, result, color, depth) in enumerate(zip(color_imgs, results, colors, depths)):
                # Mostra l'immagine a schermo (premere 'q' per uscire, anche se lo script bash lo chiuderà forzatamente)
                caps = []
                # Se è stata rilevata almeno una persona
                if result and color_img and result[0].keypoints is not None and len(result[0].keypoints.data) > 0:
                    person = result[0].keypoints.data[0].cpu().numpy()  # (17,3) -> x, y, conf
                    xy = person[:, :2]
                    conf = person[:, 2]
        
                    # Intrinseci della camera per la deproiezione
                    intr = depth.profile.as_video_stream_profile().intrinsics
                    w_img, h_img = depth.get_width(), depth.get_height()
        
                    # 1. Estrazione coordinate 3D nel frame Camera
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
        
                    # 2. Filtraggio temporale (OneEuroFilter)
                    xyz_cam_s = smoother.update(xyz_cam, conf, conf_thr)
                          
                    # 3. Trasformazione nel frame Base del Robot
                    # Usa xyz_cam_s direttamente (frame ottico nativo RealSense) invece di xyz_cam_mapped
                    xyz_base = transform_points(T_base_cam, xyz_cam_s.astype(np.float64)).astype(np.float32)
        
                    # 4. Creazione delle capsule (segmenti)
                    # (Volendo, si potrebbe considerare una logica per la quale se in questo momento non  vedo nulla, mando comunque l'ultima cosa valida, per evitare come faccio adesso di non mandare nulla.)
                    
                    # --- MODIFICA: Capsule semplificate (Braccia + Busto/Testa unico) ---
                    # Helper per validità
                    def is_valid_kpt(k):
                        return (conf[k] >= conf_thr) and np.all(np.isfinite(xyz_base[k]))
        
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
        
                # 5. Serializzazione e invio dati (impacchettamento e invio nel loop)
                t_mono_ns = time.monotonic_ns()
                # Header: Magic, Versione, Numero Capsule, Timestamp
                # struct.pack(): converte i dati in una stringa di byte secondo il formato specificato
                header = struct.pack(HDR_FMT, MAGIC, VERSION, len(caps), t_mono_ns)
                # Payload: Lista di capsule (*rec serve a spacchettare la tupla della capsula in singoli argomenti - grazie all'asterisco -)
                payload = b"".join(struct.pack(REC_FMT, *rec) for rec in caps)
                # Invio messaggio completo (header + payload) (singolo messaggio atomico)
                pub.send(header + payload)
        
                # Misura del tempo ciclo
                tNow = time.time()
                print(f"Tempo ciclo thread {n}: {tNow - t0:.3f} s")
        
                # Salvataggio video
                if save_video and video_writer is not None:
                    video_writer.write(color_img)
            
                cv2.imshow(f"YOLO Skeleton Realtime Camera {n}", color_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 
    finally:
        # Wait for both to finish
        for thread in threads:
            thread.join()

        # Cleanup risorse (fondamentale per non bloccare la RealSense al riavvio)
        print("Chiusura pipeline e finestre...")
        for pipe in pipes:
            pipe.stop()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        pub.close()
        # ctx.term()



if __name__ == "__main__":
    main()
