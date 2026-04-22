#!/usr/bin/env python3
 
"""
░█░█░█▀█░█░░░█▄█░█▀█░█▀█░░░█▀▀░▀█▀░█░░░▀█▀░█▀▀░█▀▄
░█▀▄░█▀█░█░░░█░█░█▀█░█░█░░░█▀▀░░█░░█░░░░█░░█▀▀░█▀▄
░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░▀░▀░░░▀░░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀

Merges two or more noisy arrays into a single optimal estimate
using a 1-D Kalman filter with multi-sensor sequential updates.
"""

import numpy as np
from math import pi


class KalmanFilter:
    """
    Kalman filter for human upper-body joint state estimation.

    State vector (n=65):
        [0:14]   joint positions       (p_n=14 joints)
        [14:28]  joint velocities
        [28:42]  joint accelerations
        [42:56]  joint jerks
        [56:65]  link lengths          (pl_n=9 links)

    Measurement vector (m=46):
        [0:14]   joint angles from camera 1
        [14:28]  joint angles from camera 2
        [28:37]  link lengths from camera 1
        [37:46]  link lengths from camera 2
    """

    def __init__(self):
        # Dimensions
        self.conf_thresh = 0.5
        self.n   = 3   # state vector size
        # self.m   = 46   # measurement vector size
        # self.p_n = 14   # number of joint-angle states
        # self.pl_n = 9   # number of link-length states
        # self.dt  = 0.01 # time step (s)

        # Noise parameters
        # self.var_p    = 0.05
        # self.var_pd   = 0.05
        # self.var_pdd  = 0.05
        # self.var_pddd = 0.05
        # self.psi_z    = 10.0  # base measurement noise scalar

        # Initialise all filter matrices
        self._init_state()
        # self._init_transition()
        # self._init_observation()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_state(self):
        """Initialise state vector and covariance matrices."""
        # State estimate
        self.s_k = np.zeros(self.n)
        # self.s_k[0]   = 3.0   # initial x position
        # self.s_k[56:] = np.array([0.535, 0.21, 0.355, 0.305,   # right arm link lengths
        #                            0.535, 0.21, 0.355, 0.305,   # left  arm link lengths
        #                            0.15])                        # hip link length

        # Covariance matrix
        self.p_k = np.eye(self.n) * 0.1

        # Process noise covariance Q
        self.Q = np.eye(self.n) * 1e-3

        # Measurement noise covariance R
        # self.R = np.eye(self.n)

        self.H_k = np.eye(self.n)
        self.F_k = np.eye(self.n)

    """def _init_transition(self):
        I = np.eye(self.p_n)
        Z = np.zeros((self.p_n, self.p_n))
        dt = self.dt

        # Each block row encodes:  pos, vel, acc, jerk kinematics
        F1 = np.hstack([I, dt*I, 0.5*dt**2*I, (dt**3/6)*I])
        F2 = np.hstack([Z, I,    dt*I,          0.5*dt**2*I])
        F3 = np.hstack([Z, Z,    I,             dt*I        ])
        F4 = np.hstack([Z, Z,    Z,             I           ])

        self.F_k = np.eye(self.n)
        self.F_k[0:56, 0:56] = np.vstack([F1, F2, F3, F4])

    def _init_observation(self):
        I = np.eye(self.p_n)
        self.H_k = np.zeros((self.m, self.n))

        # Both cameras observe the same joint angles (first p_n states)
        self.H_k[0:self.p_n,          0:self.p_n] = I
        self.H_k[self.p_n:2*self.p_n, 0:self.p_n] = I

        # Both cameras observe link lengths (last pl_n states)
        self.H_k[2*self.p_n:2*self.p_n+self.pl_n, -self.pl_n:] = np.eye(self.pl_n)
        self.H_k[2*self.p_n+self.pl_n:self.m,      -self.pl_n:] = np.eye(self.pl_n)"""



    # ------------------------------------------------------------------
    # Core Kalman filter steps
    # ------------------------------------------------------------------

    # Predict state and covariance one time step ahead
    def predict(self):
        self.s_k = self.F_k.dot(self.s_k)
        self.p_k = self.F_k.dot(self.p_k).dot(self.F_k.T) + self.Q

    # Correct the prediction with a new measurement
    def update(self, z_k):
        if z_k is None:
            return
        
        y_k = z_k - self.H_k.dot(self.s_k)                      # innovation
        S = self.H_k.dot(self.p_k).dot(self.H_k.T) + self.R   # innovation covariance
        K  = self.p_k.dot(self.H_k.T).dot(np.linalg.inv(S))               # Kalman gain

        self.s_k = self.s_k + K.dot(y_k)
        self.p_k = (np.eye(self.n) - K.dot(self.H_k)).dot(self.p_k)

    def filter_reset(self):
        """Reinitialise the filter to its default state (e.g. when tracking is lost)."""
        self._init_state()

    # ------------------------------------------------------------------
    # Main loop (call once per time step)
    # ------------------------------------------------------------------

    def step(self, measurement, confidence):
        self.predict()
        for z_k, conf in zip(measurement, confidence):
            if conf < self.conf_thresh:
                continue
            if np.isnan(z_k).any():
                continue
            self.R = (1-conf)**2 * np.eye(3)
            self.update(z_k)
        return self.s_k

