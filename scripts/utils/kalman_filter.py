#!/usr/bin/env python3
 
"""
░█░█░█▀█░█░░░█▄█░█▀█░█▀█░░░█▀▀░▀█▀░█░░░▀█▀░█▀▀░█▀▄
░█▀▄░█▀█░█░░░█░█░█▀█░█░█░░░█▀▀░░█░░█░░░░█░░█▀▀░█▀▄
░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░▀░▀░░░▀░░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀

Merges two or more noisy arrays into a single optimal estimate
using a 1-D Kalman filter with multi-sensor sequential updates.
"""

import numpy as np


class KalmanFilter:
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

