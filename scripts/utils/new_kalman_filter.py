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
        self.n   = 65   # state vector size
        self.m   = 46   # measurement vector size
        self.p_n = 14   # number of joint-angle states
        self.pl_n = 9   # number of link-length states
        self.dt  = 0.01 # time step (s)

        # Noise parameters
        self.var_p    = 0.05
        self.var_pd   = 0.05
        self.var_pdd  = 0.05
        self.var_pddd = 0.05
        self.psi_z    = 10.0  # base measurement noise scalar

        # Initialise all filter matrices
        self._init_state()
        self._init_transition()
        self._init_observation()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_state(self):
        """Initialise state vector and covariance matrices."""
        # State estimate
        self.s_k = np.zeros(self.n)
        self.s_k[0]   = 3.0   # initial x position
        self.s_k[56:] = np.array([0.535, 0.21, 0.355, 0.305,   # right arm link lengths
                                   0.535, 0.21, 0.355, 0.305,   # left  arm link lengths
                                   0.15])                        # hip link length

        # Covariance matrix
        self.p_k = np.eye(self.n)

        # Process noise covariance Q
        self.Q = np.diag(np.hstack([
            np.ones(self.p_n) * (self.var_p    ** 2) * (self.dt ** 4) / 24,
            np.ones(self.p_n) * (self.var_pd   ** 2) * (self.dt ** 3) / 6,
            np.ones(self.p_n) * (self.var_pdd  ** 2) * (self.dt ** 2) / 2,
            np.ones(self.p_n) * (self.var_pddd ** 2) *  self.dt,
            np.zeros(self.pl_n)
        ]))

        # Measurement noise covariance R
        self.R = self.psi_z * np.eye(self.m)

    def _init_transition(self):
        """Build the constant-jerk state-transition matrix F_k."""
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
        """Build the observation matrix H_k."""
        I = np.eye(self.p_n)
        self.H_k = np.zeros((self.m, self.n))

        # Both cameras observe the same joint angles (first p_n states)
        self.H_k[0:self.p_n,          0:self.p_n] = I
        self.H_k[self.p_n:2*self.p_n, 0:self.p_n] = I

        # Both cameras observe link lengths (last pl_n states)
        self.H_k[2*self.p_n:2*self.p_n+self.pl_n, -self.pl_n:] = np.eye(self.pl_n)
        self.H_k[2*self.p_n+self.pl_n:self.m,      -self.pl_n:] = np.eye(self.pl_n)

    # ------------------------------------------------------------------
    # Core Kalman filter steps
    # ------------------------------------------------------------------

    def predict(self):
        """Step 1 – Predict state and covariance one time step ahead."""
        self.s_k = self.F_k.dot(self.s_k)
        self.p_k = self.F_k.dot(self.p_k).dot(self.F_k.T) + self.Q

    def update(self, z_k, no_new1: bool, no_new2: bool):
        """
        Step 2 – Correct the prediction with a new measurement.

        Parameters
        ----------
        z_k     : array (m,) – stacked measurement vector from both cameras.
        no_new1 : bool – True when camera 1 has no fresh data this cycle.
        no_new2 : bool – True when camera 2 has no fresh data this cycle.
        """
        if z_k is None:
            return

        # Only update when at least one camera provides new data
        if not (no_new1 and no_new2):
            y_k   = z_k - self.H_k.dot(self.s_k)                      # innovation
            S     = self.H_k.dot(self.p_k).dot(self.H_k.T) + self.R   # innovation covariance
            K  = self.p_k.dot(self.H_k.T).dot(np.linalg.inv(S))               # Kalman gain

            self.s_k = self.s_k + K.dot(y_k)
            self.p_k = (np.eye(self.n) - K.dot(self.H_k)).dot(self.p_k)

    def compute_bounds(self, alpha, dt):
        """
        Step 3 – Clamp state, velocity and acceleration to physiological limits.

        Parameters
        ----------
        alpha : array (n,) – current state vector (modified in-place).
        dt    : float      – time step used for derivative bound propagation.

        Returns
        -------
        s_kb  : array (n,) – saturated state vector.
        """
        plink_lim = [550, 250, 370, 350, 550, 250, 370, 350, 150]  # mm -> converted below

        # [upper, lower] bounds for each of the 14 joint angles (rad)
        alpha_b = [
            [ np.inf,          -np.inf         ],  # 0  x hip
            [ np.inf,          -np.inf         ],  # 1  y hip
            [ 2.0,             -2.0            ],  # 2  z hip
            [ np.pi,           -np.pi          ],  # 3  heading
            [ np.pi/2,         -np.pi/6        ],  # 4  tau_r
            [ 8*np.pi/9,       -np.pi/20       ],  # 5  Alpha1_r
            [ 0,                0              ],  # 6  Alpha2_r  (set dynamically)
            [ 0,                0              ],  # 7  Alpha3_r  (set dynamically)
            [ np.pi/2,         -70*np.pi/180   ],  # 8  Alpha4_r
            [ np.pi/2,         -np.pi/6        ],  # 9  tau_l
            [ 8*np.pi/9,       -np.pi/20       ],  # 10 Alpha1_l
            [ 0,                0              ],  # 11 Alpha2_l  (set dynamically)
            [ 0,                0              ],  # 12 Alpha3_l  (set dynamically)
            [ np.pi/2,         -70*np.pi/180   ],  # 13 Alpha4_l
        ]

        # Velocity bounds
        a_d_b = [
            [ 0.8,           0            ], [ 0.8,       -0.8        ], [ 0.1,       -0.1       ], [ np.pi/4,  -np.pi/4  ],
            [ np.pi/10,     -np.pi/10     ], [ np.pi/2,  -np.pi/2    ], [ np.pi/2,  -np.pi/2   ], [ np.pi/2,  -np.pi/2  ], [ 3*np.pi/10, -3*np.pi/10 ],
            [ np.pi/10,     -np.pi/10     ], [ np.pi/2,  -np.pi/2    ], [ np.pi/2,  -np.pi/2   ], [ np.pi/2,  -np.pi/2  ], [ 3*np.pi/10, -3*np.pi/10 ],
        ]

        # Acceleration bounds
        a_dd_b = [
            [ 0.1,          -0.1          ], [ 0.1,       -0.1       ], [ 0.01,     -0.01      ], [ np.pi/8,  -np.pi/8  ],
            [ np.pi/20,     -np.pi/20     ], [ np.pi/4,  -np.pi/4   ], [ np.pi/4,  -np.pi/4   ], [ np.pi/4,  -np.pi/4  ], [ 3*np.pi/20, -3*np.pi/20 ],
            [ np.pi/20,     -np.pi/20     ], [ np.pi/4,  -np.pi/4   ], [ np.pi/4,  -np.pi/4   ], [ np.pi/4,  -np.pi/4  ], [ 3*np.pi/20, -3*np.pi/20 ],
        ]

        # --- Dynamic coupling: Alpha2 and Alpha3 bounds depend on Alpha1 ---

        # Left arm
        a_1l = np.clip(alpha[10], alpha_b[10][1], alpha_b[10][0])
        a_2l = alpha[11]
        alpha_b[11] = [np.deg2rad(153) - a_1l / 6,   np.deg2rad(-43) + a_1l / 3]
        a_2l = np.clip(a_2l, alpha_b[11][1], alpha_b[11][0])
        alpha_b[12] = [
            np.deg2rad(60)  + 4*a_1l/9 - 5*a_2l/9 + 5*a_1l*a_2l/810,
            -np.pi/2        + 7*a_1l/9 -   a_2l/9 + 2*a_1l*a_2l/810,
        ]

        # Right arm
        a_1r = np.clip(alpha[5], alpha_b[5][1], alpha_b[5][0])
        a_2r = alpha[6]
        alpha_b[6] = [np.deg2rad(153) - a_1r / 6,   np.deg2rad(-43) + a_1r / 3]
        a_2r = np.clip(a_2r, alpha_b[6][1], alpha_b[6][0])
        alpha_b[7] = [
            np.deg2rad(60)  + 4*a_1r/9 - 5*a_2r/9 + 5*a_1r*a_2r/810,
            -np.pi/2        + 7*a_1r/9 -   a_2r/9 + 2*a_1r*a_2r/810,
        ]

        # --- Saturate positions, velocities, accelerations ---
        s_kb = alpha.copy()
        for i in range(14):
            hi, lo = alpha_b[i]

            s_kb[i] = np.clip(alpha[i], lo, hi)

            vel_hi = min(a_d_b[i][0],  (hi - alpha[i]) / dt)
            vel_lo = max(a_d_b[i][1],  (lo - alpha[i]) / dt)
            s_kb[i + 14] = np.clip(alpha[i + 14], vel_lo, vel_hi)

            acc_hi = min(a_dd_b[i][0], (a_d_b[i][0] - alpha[i + 14]) / dt)
            acc_lo = max(a_dd_b[i][1], (a_d_b[i][1] - alpha[i + 14]) / dt)
            s_kb[i + 28] = np.clip(alpha[i + 28], acc_lo, acc_hi)

        # --- Clamp link lengths ---
        for j in range(9):
            jj = 56 + j
            if s_kb[jj] > plink_lim[j] / 1000:
                s_kb[jj] = plink_lim[j] / 1000

        return s_kb

    def filter_reset(self):
        """Reinitialise the filter to its default state (e.g. when tracking is lost)."""
        self._init_state()

    # ------------------------------------------------------------------
    # Main loop (call once per time step)
    # ------------------------------------------------------------------

    def step(self, z_k, R_diag_cam1=None, R_diag_cam2=None,
             no_new1: bool = False, no_new2: bool = False):
        """
        Run one complete predict → update → bound-clamp cycle.

        Parameters
        ----------
        z_k          : array (46,) – measurement vector [cam1_angles(14),
                                     cam2_angles(14), cam1_links(9), cam2_links(9)].
        R_diag_cam1  : array (14,) or None – per-joint measurement noise for cam 1.
        R_diag_cam2  : array (14,) or None – per-joint measurement noise for cam 2.
        no_new1/2    : bool – flags indicating stale camera data.

        Returns
        -------
        s_k : array (65,) – filtered state vector after this step.
        """
        # Optionally update measurement noise from camera confidence scores
        if R_diag_cam1 is not None:
            self.R[0:self.p_n, 0:self.p_n] = np.diag(R_diag_cam1)
        if R_diag_cam2 is not None:
            self.R[self.p_n:2*self.p_n, self.p_n:2*self.p_n] = np.diag(R_diag_cam2)

        self.predict()
        self.update(z_k, no_new1, no_new2)
        # self.s_k = self.compute_bounds(self.s_k, self.dt)
        return self.s_k


# ----------------------------------------------------------------------
# Minimal usage example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    kf = KalmanFilter()

    # Simulate 100 steps with a dummy measurement vector
    for step in range(100):
        z_k = np.random.randn(46) * 0.1  # replace with real measurements
        state = kf.step(z_k)
        print(f"Step {step:3d} | hip_x={state[0]:.4f}  hip_y={state[1]:.4f}")