#!/usr/bin/env python3

"""
░█▀▀░█▀█░█░░░▀█▀░█▀▄░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
░█░░░█▀█░█░░░░█░░█▀▄░█▀▄░█▀█░░█░░░█░░█░█░█░█
░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀
"""

import numpy as np
import cv2
import os
from utils.skeleton_tracker import SkeletonTracker
import pyrealsense2 as rs
from utils.marker_detector  import * # MarkerDetector

# Parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)


# Function for saving rotation matrix on file
def write_rotation_matrix_to_file(file, mat):
    with open(file, 'w') as file:
        for i in range(4):
            file.write('\t'.join(map(str, mat[i, :])) + '\n')


def saveCoefficients(mtx, dist):
    cv_file = cv2.FileStorage("calib_images/calibrationCoefficients.yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def loadCoefficients():
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage("calib_images/calibrationCoefficients.yaml", cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("camera_matrix").mat()
    dist_matrix = cv_file.getNode("dist_coeff").mat()

    # Debug: print the values
    # print("camera_matrix : ", camera_matrix.tolist())
    # print("dist_matrix : ", dist_matrix.tolist())

    cv_file.release()
    return [camera_matrix, dist_matrix]


# Function to ensure that average matrix is correctly defined (det(R)=1)
def correct_rotation_matrix(rotation_matrix):
    rot = rotation_matrix[:3, :3]
    U, _, Vt = np.linalg.svd(rot)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1   # flip last column
        R = np.dot(U, Vt)
    rotation_matrix[:3, :3] = R
    return rotation_matrix


# Main loop of marker detection and rotation matrix creation
def main():
    # Create pipeline and start config
    align = rs.align(rs.stream.color) # Allinea depth a color
    ctx = rs.context()
    devices = ctx.devices  # Query connected devices

    print("Press 'y' to accept the proposed reference system, otherwise press 'n' to skip")
    
    for i, device in enumerate(devices):
        tracker = SkeletonTracker(device.get_info(rs.camera_info.serial_number), 1920, 1080, 30, False)
        mark = MarkerDetector(tracker)

        rotation_matrix = mark.simple_calibration(34)
        serial = device.get_info(rs.camera_info.serial_number)
        save_file = os.path.join(data_dir, f"calibration/pose_{serial}.txt")
        matrix = correct_rotation_matrix(rotation_matrix)
        write_rotation_matrix_to_file(save_file, matrix)

        # rotation_matrices = []
        # for marker_ID in marker_IDs:
        #     serial = device.get_info(rs.camera_info.serial_number)
        #     rotation_matrix = mark.static_calibration(marker_ID)
        #     if not rotation_matrix is None:
        #         rotation_matrices.append(rotation_matrix)
# 
        # if len(rotation_matrices) == 0:
        #     print(f"Calibration failed for device {i} (SN: {serial}). No marker detected.")
        #     return
        # elif len(rotation_matrices) > 0 and len(rotation_matrices) <= len(marker_IDs):
        #     avg_rotation_matrix = np.mean([rotation_matrix for rotation_matrix in rotation_matrices], axis=0)
# 
        #     print(f"Media: {avg_rotation_matrix}")
        #     matrix = correct_rotation_matrix(avg_rotation_matrix)
# 
        #     print(f"Finl: {matrix}")
# 
        #     script_dir = os.path.dirname(os.path.abspath(__file__))
        #     save_file = os.path.join(data_dir, f"calibration/pose_{serial}.txt")
        #     write_rotation_matrix_to_file(save_file, matrix)

    print(f"Calibration ended correctly. Marker was detected by all the devices.")


# Entry point
if __name__ == '__main__':
    main()
    

    