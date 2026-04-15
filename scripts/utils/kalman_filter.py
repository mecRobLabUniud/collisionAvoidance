#!/usr/bin/env python3
 
"""
░█░█░█▀█░█░░░█▄█░█▀█░█▀█░░░█▀▀░▀█▀░█░░░▀█▀░█▀▀░█▀▄
░█▀▄░█▀█░█░░░█░█░█▀█░█░█░░░█▀▀░░█░░█░░░░█░░█▀▀░█▀▄
░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░▀░▀░░░▀░░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀

Merges two or more noisy arrays into a single optimal estimate
using a 1-D Kalman filter with multi-sensor sequential updates.
"""

from typing import Dict, List, Optional, Tuple



class KalmanFilter():
    def __init__( self, process_noise: float = 0.01, measurement_noise: float = 5.0, initial_estimate: float = 0.0, initial_covariance: float = 1.0):
        if process_noise <= 0:
            raise ValueError("process_noise (Q) must be positive.")
        if measurement_noise <= 0:
            raise ValueError("measurement_noise (R) must be positive.")
        if initial_covariance <= 0:
            raise ValueError("initial_covariance (P0) must be positive.")
 
        self.Q = process_noise
        self.R = measurement_noise
        self.x0 = initial_estimate
        self.P0 = initial_covariance
 
        # Runtime state (populated after merge)
        self._estimates: List[float] = []
        self._gains: List[float] = []
        self._covariances: List[float] = []
 
    def merge(self, *arrays: List[float]) -> List[float]:
        if len(arrays) < 2:
            raise ValueError("Provide at least two arrays to merge.")
 
        n = max(len(a) for a in arrays)
 
        x = self.x0
        P = self.P0
        estimates, gains, covariances = [], [], []
 
        for t in range(n):
            # --- Predict step ---
            P = P + self.Q
 
            # --- Update step (sequential, one sensor at a time) ---
            step_gain = 0.0
            observations = [a[t] for a in arrays if t < len(a)]
 
            for z in observations:
                K = P / (P + self.R)
                x = x + K * (z - x)
                P = (1.0 - K) * P
                step_gain = K  # keep last gain for this timestep
 
            estimates.append(round(x, 6))
            gains.append(round(step_gain, 6))
            covariances.append(round(P, 6))
 
        self._estimates = estimates
        self._gains = gains
        self._covariances = covariances
        return estimates
 
    def merge_single_step( self, observations: List[float], x: Optional[float] = None, P: Optional[float] = None) -> Tuple[float, float]:
        x = self.x0 if x is None else x
        P = self.P0 if P is None else P
 
        P += self.Q
        for z in observations:
            K = P / (P + self.R)
            x = x + K * (z - x)
            P = (1.0 - K) * P

        return x, P
 
    @property
    def estimates(self) -> List[float]:
        return self._estimates

    @property
    def kalman_gains(self) -> List[float]:
        return self._gains

    @property
    def covariances(self) -> List[float]:
        return self._covariances

    @property
    def stats(self) -> Dict:
        if not self._estimates:
            return {}
        n = len(self._estimates)
        mean = sum(self._estimates) / n
        variance = sum((v - mean) ** 2 for v in self._estimates) / n
        return {
            "length": n,
            "mean": round(mean, 4),
            "variance": round(variance, 4),
            "std_dev": round(variance ** 0.5, 4),
            "min": min(self._estimates),
            "max": max(self._estimates),
            "avg_kalman_gain": round(sum(self._gains) / len(self._gains), 4),
        }

    def __repr__(self) -> str:
        return (
            f"KalmanFilter("
            f"Q={self.Q}, R={self.R}, "
            f"x0={self.x0}, P0={self.P0})"
        )