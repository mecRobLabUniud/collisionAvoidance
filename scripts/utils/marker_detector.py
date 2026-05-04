#!/usr/bin/env python3

"""
░█▄█░█▀█░█▀▄░█░█░█▀▀░█▀▄░░░█▀▄░█▀▀░▀█▀░█▀▀░█▀▀░▀█▀░█▀█░█▀▄
░█░█░█▀█░█▀▄░█▀▄░█▀▀░█▀▄░░░█░█░█▀▀░░█░░█▀▀░█░░░░█░░█░█░█▀▄
░▀░▀░▀░▀░▀░▀░▀░▀░▀▀▀░▀░▀░░░▀▀░░▀▀▀░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀
"""

import numpy as np
import cv2
import cv2.aruco as aruco

# Parameters
marker_IDs = [17, 21 , 34, 42, 50]
pitch = 0.029
transformations = [np.eye(4) for _ in marker_IDs]   # transformation matrices from any marker frame to marker 34 frame (taken as the reference one)
transformations[0] = np.array([[0, -1, 0, 0],
                               [0, 0, -1, -pitch],
                               [1, 0, 0, -pitch],
                               [0, 0, 0, 1]], dtype=np.float32)
transformations[1] = np.array([[0, 0, -1, -pitch],
                               [0, 1, 0, 0],
                               [1, 0, 0, -pitch],
                               [0, 0, 0, 1]], dtype=np.float32)
transformations[3] = np.array([[0, 0, 1, pitch],
                               [0, -1, 0, 0],
                               [1, 0, 0, -pitch],
                               [0, 0, 0, 1]], dtype=np.float32)
transformations[4] = np.array([[0, 1, 0, 0],
                               [0, 0, 1, pitch],
                               [1, 0, 0, -pitch],
                               [0, 0, 0, 1]], dtype=np.float32)


class MarkerDetector:
    def __init__(self, tracker):
        self.dim = 0.05
        self.tracker = tracker
        self.matrix_coefficients, self.distortion_coefficients = tracker.get_intrinsics()
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def calibration(self, marker_ID): 
        while True:   
            # operations on the frame come here
            frame = self.tracker.get_color_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
            dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            parameters = aruco.DetectorParameters()  # new style
            detector = aruco.ArucoDetector(dictionary, parameters)

            # lists of ids and the corners beloning to each id
            corners, ids, _ = detector.detectMarkers(gray)

            rotation_matrix = None
            if np.all(ids is not None):
                zipped = zip(ids, corners)
                ids, corners = zip(*(sorted(zipped)))
                axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
                # Estimate pose of each marker
                for i in range(len(ids)):
                    if ids[i] == marker_ID:
                        rot, pos, _ = aruco.estimatePoseSingleMarkers(corners[i], self.dim, self.matrix_coefficients, self.distortion_coefficients)

                        # Build 4x4 pose matrix [R | t; 0 0 0 1]
                        R_mat, _ = cv2.Rodrigues(rot)
                        rotation_matrix = np.eye(4, dtype=np.float32)
                        rotation_matrix[:3, :3] = R_mat  # Rotation part
                        rotation_matrix[:3, 3] = pos.flatten()  # Translation part

                        # rotation_matrix = np.linalg.inv(rotation_matrix)
                        # aa = 
                        rotation_matrix = np.dot(transformations[marker_IDs.index(marker_ID)], np.linalg.inv(rotation_matrix))

                        aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                        imgpts, jac = cv2.projectPoints(axis, rot, pos, self.matrix_coefficients,
                                                        self.distortion_coefficients)

                        cv2.drawFrameAxes(frame, self.matrix_coefficients, self.distortion_coefficients, rot, pos, length=0.1)
                        relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                        cv2.circle(frame, relativePoint, 2, (255, 255, 0))
           
            # Display the resulting frame
            cv2.imshow('frame', frame)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                return rotation_matrix