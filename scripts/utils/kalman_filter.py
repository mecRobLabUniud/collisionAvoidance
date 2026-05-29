#!/usr/bin/env python3
 
"""
░█░█░█▀█░█░░░█▄█░█▀█░█▀█░░░█▀▀░▀█▀░█░░░▀█▀░█▀▀░█▀▄
░█▀▄░█▀█░█░░░█░█░█▀█░█░█░░░█▀▀░░█░░█░░░░█░░█▀▀░█▀▄
░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░▀░▀░░░▀░░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀

Merges two or more noisy arrays into a single optimal estimate
using a 3-D or 6-D Kalman filter with multi-sensor sequential updates.
"""

import numpy as np




class SimpleMerger:
    def __init__(self):
        pass

    # Main loop for predicting and updating the filter with new measurements
    def step(self, measurement, confidence):
        merge = []
        for meas in zip(*measurement):
            merge.append(sum(m*c for m, c in zip(meas, confidence)if not np.isnan(m))/(sum(confidence)))
        return np.array(merge) if len(measurement) > 0 else np.array([np.nan, np.nan, np.nan])
    


class KalmanFilter3D:
    def __init__(self):
        self.maha_thr = 9.0
        self.n   = 3
        self.s_k = np.zeros(self.n)
        self.p_k = np.eye(self.n) * 0.1
        self.Q = np.eye(self.n) * 1e-3
        self.H_k = np.eye(self.n)
        self.F_k = np.eye(self.n)

    # Predict state and covariance one time step ahead
    def predict(self):
        self.s_k = self.F_k.dot(self.s_k)
        self.p_k = self.F_k.dot(self.p_k).dot(self.F_k.T) + self.Q

    # Correct the prediction with a new measurement
    def update(self, z_k):
        if z_k is None:
            return 1
        
        y_k = z_k - self.H_k.dot(self.s_k)
        S = self.H_k.dot(self.p_k).dot(self.H_k.T) + self.R
        d = self.mahalanobis_distance(y_k, S)
        if d > self.maha_thr:
            return 1
        K  = self.p_k.dot(self.H_k.T).dot(np.linalg.inv(S))

        self.s_k = self.s_k + K.dot(y_k)
        self.p_k = (np.eye(self.n) - K.dot(self.H_k)).dot(self.p_k)
        return 0

    # Reinitialise the filter to its default state
    def filter_reset(self):
        self._init_state()

    # Select outliersbased on Mahalanobis distance
    def mahalanobis_distance(self, y_k, S):
        invS = np.linalg.inv(S)
        d = y_k.dot(invS).dot(y_k)
        return d

    # Main loop for predicting and updating the filter with new measurements
    def step(self, measurement, confidence):
        self.predict()
        updated = False
        for z_k, conf in zip(measurement, confidence):
            if np.isnan(z_k).any():
                continue
            self.R = (1.1-conf)**2 * np.eye(3)    # modified 1 to 1.1
            res = self.update(z_k)
            if res:
                continue
            updated = True
        
        return self.s_k if updated else np.array([np.nan, np.nan, np.nan])











class KalmanFilter6D:
    def __init__(self):
        self.conf_thresh = 0.5
        self.maha_thresh = 9.0
        self.n   = 6
        self.dt = 50.0
        self.s_k = np.zeros(self.n)
        self.old_meas = np.zeros(int(self.n/2))
        self.p_k = np.eye(self.n) * 0.1
        self.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
        self.H_k = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0]])
        self.F_k = np.array([[1, 0, 0, self.dt, 0,  0 ],
                            [0, 1, 0, 0,  self.dt, 0 ],
                            [0, 0, 1, 0,  0,  self.dt],
                            [0, 0, 0, 1,  0,  0 ],
                            [0, 0, 0, 0,  1,  0 ],
                            [0, 0, 0, 0,  0,  1 ]])

    # Predict state and covariance one time step ahead
    def predict(self):
        self.s_k = self.F_k.dot(self.s_k)
        self.p_k = self.F_k.dot(self.p_k).dot(self.F_k.T) + self.Q

    # Correct the prediction with a new measurement
    def update(self, z_k):
        if z_k is None:
            return 1
        
        y_k = z_k - self.H_k.dot(self.s_k)
        S = self.H_k.dot(self.p_k).dot(self.H_k.T) + self.R
        d = self.mahalanobis_distance(y_k, S)
        if d > self.maha_thresh:
            return 1
        K  = self.p_k.dot(self.H_k.T).dot(np.linalg.inv(S))

        self.s_k = self.s_k + K.dot(y_k)
        self.p_k = (np.eye(self.n) - K.dot(self.H_k)).dot(self.p_k)
        return 0

    # Reinitialise the filter to its default state
    def filter_reset(self):
        self._init_state()

    # Select outliersbased on Mahalanobis distance
    def mahalanobis_distance(self, y_k, S):
        invS = np.linalg.inv(S)
        d = y_k.dot(invS).dot(y_k)
        return d

    # Main loop for predicting and updating the filter with new measurements
    def step(self, measurement, confidence):
        self.predict()
        updated = False
        for meas, conf in zip(measurement, confidence):
            z_k = meas # np.array(np.hstack([meas, (meas - self.old_meas) / self.dt ]))

            if conf < self.conf_thresh:
                continue
            if np.isnan(z_k).any():
                continue
            self.R = (1.1-conf)**2 * np.eye(int(self.n/2))  # modified 1 to 1.1
            res = self.update(z_k)
            if res:
                continue
            updated = True
        
        self.old_meas = self.s_k[:3].copy()
        
        return self.s_k[:3] if updated else np.array([np.nan, np.nan, np.nan])










def _joseph_update(P, K, H, n):
    """Numerically stable Joseph-form covariance update.
    P_new = (I - KH) P (I - KH)^T + K R K^T
    """
    I_KH = np.eye(n) - K.dot(H)
    return I_KH.dot(P).dot(I_KH.T)


class _CameraOcclusionState:
    """Simple state machine that flags a camera as occluded when its
    measurements imply an impossible instantaneous speed.
 
    States
    ------
    NOMINAL  : measurements accepted normally
    OCCLUDED : measurements heavily down-weighted (R inflated)
 
    Transitions
    -----------
    NOMINAL  -> OCCLUDED : speed > max_speed on any single frame
    OCCLUDED -> NOMINAL  : `recovery_frames` consecutive frames with
                           speed <= max_speed
    """
 
    def __init__(self, max_speed: float, recovery_frames: int = 5,
                 occlusion_R_scale: float = 50.0):
        self.max_speed       = max_speed          # m / (filter time-step)
        self.recovery_frames = recovery_frames
        self.occ_R_scale     = occlusion_R_scale  # how much to inflate R when occluded
        self._occluded       = False
        self._ok_streak      = 0
        self._last_pos       = None               # last accepted position
 
    @property
    def is_occluded(self):
        return self._occluded
 
    def update(self, z_k: np.ndarray) -> bool:
        """Feed a new position measurement.  Returns True if the
        measurement looks valid (not an occlusion jump)."""
        if self._last_pos is None:
            self._last_pos = z_k.copy()
            return True
 
        speed = float(np.linalg.norm(z_k - self._last_pos))
        jump  = speed > self.max_speed
 
        if jump:
            self._occluded  = True
            self._ok_streak = 0
            # do NOT update _last_pos — keep the last good position
            return False
        else:
            if self._occluded:
                self._ok_streak += 1
                if self._ok_streak >= self.recovery_frames:
                    self._occluded  = False
                    self._ok_streak = 0
            self._last_pos = z_k.copy()
            return True
 
    def reset(self):
        self._occluded  = False
        self._ok_streak = 0
        self._last_pos  = None


class ImprovedKalmanFilter6D:
    """6-D Kalman filter fusing N camera sources.
 
    State vector: [x, y, z, vx, vy, vz]
 
    Occlusion handling
    ------------------
    1. **Velocity gate** – if the implied speed from the previous
       accepted position exceeds `max_speed_m_per_s` the measurement
       is flagged as an occlusion artefact and its R is inflated by
       `occlusion_R_scale` (instead of being accepted at face value).
 
    2. **Adaptive Mahalanobis gate** – the threshold is widened by
       sqrt(trace(P)/n) when the filter is uncertain, allowing valid
       re-acquisitions after long occlusions while still rejecting
       wild outliers.
 
    3. **Per-camera state machine** – tracks consecutive jump frames
       per camera so that one bad frame does not permanently blacklist
       a camera, but also does not let a stream of bad frames slip
       through.
 
    Parameters
    ----------
    dt : float
        Time step in seconds (default 0.05 → 20 Hz).
    max_speed_m_per_s : float
        Maximum plausible speed in metres per second (default 3 m/s,
        roughly a fast walk; increase for sports / vehicles).
    conf_thresh : float
        Minimum detector confidence to even consider a measurement.
    maha_thresh : float
        Base Mahalanobis distance threshold.
    recovery_frames : int
        Consecutive good frames required to exit OCCLUDED state.
    occlusion_R_scale : float
        How much to inflate R when a camera is flagged as occluded.
    """
 
    def __init__(
        self,
        dt: float = 0.05,
        max_speed_m_per_s: float = 3.0,
        conf_thresh: float = 0.5,
        maha_thresh: float = 9.0,
        recovery_frames: int = 5,
        occlusion_R_scale: float = 50.0,
    ):
        self.conf_thresh      = conf_thresh
        self.maha_thresh      = maha_thresh
        self.occlusion_R_scale = occlusion_R_scale
 
        self.n  = 6
        self.dt = dt
 
        self.s_k = np.zeros(self.n)
        self.p_k = np.eye(self.n) * 0.1
 
        # Process noise — position much smaller than velocity noise
        self.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
 
        # Observation matrix  (we only measure position)
        self.H_k = np.zeros((3, 6))
        self.H_k[:, :3] = np.eye(3)
 
        # State transition matrix
        self.F_k = np.eye(self.n)
        self.F_k[0, 3] = dt
        self.F_k[1, 4] = dt
        self.F_k[2, 5] = dt
 
        self.R = np.eye(3) * 0.1   # updated per measurement in step()
 
        # Per-camera occlusion trackers — created lazily in step()
        self._max_speed_per_step = max_speed_m_per_s * dt
        self._recovery_frames    = recovery_frames
        self._cam_states: list[_CameraOcclusionState] = []
 
    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------
 
    def predict(self):
        self.s_k = self.F_k.dot(self.s_k)
        self.p_k = self.F_k.dot(self.p_k).dot(self.F_k.T) + self.Q
 
    def update(self, z_k: np.ndarray, adaptive_gate: bool = True) -> int:
        """Return 0 on success, 1 if measurement rejected."""
        if z_k is None:
            return 1
 
        y_k = z_k - self.H_k.dot(self.s_k)
        S   = self.H_k.dot(self.p_k).dot(self.H_k.T) + self.R
        d   = self.mahalanobis_distance(y_k, S)
 
        # Adaptive gate: relax threshold when filter is uncertain
        threshold = self.maha_thresh
        if adaptive_gate:
            uncertainty_scale = np.sqrt(np.trace(self.p_k) / self.n)
            threshold = self.maha_thresh * max(1.0, uncertainty_scale)
 
        if d > threshold:
            return 1
 
        K = self.p_k.dot(self.H_k.T).dot(np.linalg.inv(S))
        self.s_k = self.s_k + K.dot(y_k)
        self.p_k = _joseph_update(self.p_k, K, self.H_k, self.n)    #!!!
        return 0
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def filter_reset(self):
        """Reinitialise the filter and all per-camera states."""
        self.s_k = np.zeros(self.n)
        self.p_k = np.eye(self.n) * 0.1
        for cam in self._cam_states:
            cam.reset()
 
    def mahalanobis_distance(self, y_k: np.ndarray, S: np.ndarray) -> float:
        invS = np.linalg.inv(S)
        return float(y_k.dot(invS).dot(y_k))
 
    def step(self, measurement: list, confidence: list) -> np.ndarray:
        """Predict + sequential update from all cameras.
 
        Parameters
        ----------
        measurement : list of np.ndarray  shape (3,)
            One 3-D position per camera; use np.full(3, np.nan) for missing.
        confidence  : list of float
            Detector confidence in [0, 1] for each camera.
 
        Returns
        -------
        np.ndarray shape (3,)
            Fused position estimate, or [nan, nan, nan] if no valid update.
        """
        # Grow per-camera state list lazily
        n_cams = len(measurement)
        while len(self._cam_states) < n_cams:
            self._cam_states.append(
                _CameraOcclusionState(
                    max_speed=self._max_speed_per_step,
                    recovery_frames=self._recovery_frames,
                    occlusion_R_scale=self.occlusion_R_scale,
                )
            )
 
        self.predict()
        updated = False
 
        for i, (meas, conf) in enumerate(zip(measurement, confidence)):
            # --- basic validity checks ---
            if conf < self.conf_thresh:
                continue
            if np.isnan(meas).any():
                continue
 
            z_k    = np.asarray(meas, dtype=float)
            cam_st = self._cam_states[i]
 
            # --- occlusion / jump detection ---
            meas_ok = cam_st.update(z_k)
 
            # Base measurement noise from confidence
            r_base = (1.1 - conf) ** 2
 
            # if not meas_ok or cam_st.is_occluded:
            #     # Inflate R heavily — the measurement may be garbage,
            #     # but we still pass it so the filter stays numerically
            #     # warm.  The inflated R means it will have minimal effect.
            #     r_scale = self.occlusion_R_scale
            # else:
            #     r_scale = 1.0

            r_scale = 1.0
 
            self.R = r_base * r_scale * np.eye(3)
 
            res = self.update(z_k)
            if res == 0:
                updated = True
 
        return self.s_k[:3].copy() if updated else np.full(3, np.nan)