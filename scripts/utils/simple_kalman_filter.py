import numpy as np

def setup_kalman_3d_simple():
    # Stato x = [px, py, pz]
    x = np.zeros(3)                 # posizione iniziale
    P = np.eye(3) * 0.1             # covarianza iniziale (piccola)

    # Rumore di processo (posizione, solo varianza di posizione)
    Q = np.eye(3) * 1e-3            # da tuning: rumore intrinseco del processo

    return x, P, Q

def kalman_predict_simple(x, P, Q):
    # In questo caso: x_{k+1|k} = x_k (modello “statico” o “quasi‑inerziale”)
    x_pred = x.copy()
    P_pred = P + Q
    return x_pred, P_pred

def kalman_update_simple(x_pred, P_pred, z, R):
    """
    x_pred: stato predetto (3,)
    P_pred: covarianza predetta (3,3)
    z:      misura da una cam (3,)
    R:      covarianza di misura (3,3)
    """
    H = np.eye(3)  # z = H x + v

    # Innovazione
    y = z - H @ x_pred               # (3,)

    # Matrice di covarianza dell'innovazione
    S = H @ P_pred @ H.T + R         # (3,3)

    # Guadagno di Kalman
    K = P_pred @ H.T @ np.linalg.inv(S)  # (3,3)

    # Aggiornamento
    x_upd = x_pred + K @ y
    P_upd = (np.eye(3) - K @ H) @ P_pred

    return x_upd, P_upd

def distanza_mahalanobis(y, S):
    """y = innovazione (3,), S = covarianza (3,3)"""
    invS = np.linalg.inv(S)
    d2 = y @ invS @ y
    return d2

# Parametri
th_maha = 9.0       # soglia chi‑quadrato 3D (circa 99%)
conf_thresh = 0.5   # soglia minima di confidence per usare una cam



def simple_kalman_filter(measurements, conf):

    # Inizializza Kalman
    x, P, Q = setup_kalman_3d_simple()

    # 1. PREDIZIONE
    x_pred, P_pred = kalman_predict_simple(x, P, Q) # s_k, p_k


    # 2. SELEZIONE + AGGIORNAMENTO (per cam accettate)
    for meas, conf in zip(measurements, conf):
        z = meas
        R = (1-conf)**2 * np.eye(3)
        conf = conf

        if np.isnan(meas).any():
            continue

        # 1. Prima: requito di confidence
        # if conf < conf_thresh:
        #     continue

        # 2. Calcola innovazione e distanza Mahalanobis
        z_pred = np.eye(3) @ x_pred
        y = z - z_pred
        S = np.eye(3) @ P_pred @ np.eye(3).T + R
        d2 = distanza_mahalanobis(y, S)

        # if d2 > th_maha:
        #     continue  # outlier

        # 3. Aggiorna con questa misura
        x, P = kalman_update_simple(x_pred, P_pred, z, R)

        # Aggiorna predizione per il prossimo aggiornamento
        x_pred = x  # s_k
        P_pred = P  # p_k

    # 3. USCITA: stima posizione 3D unica
    p_fused = x          # posizione filtrata 3D

    return p_fused