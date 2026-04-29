"""import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the dictionary we want to use
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate a marker
marker_id = 42
marker_size = 200  # Size in pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

cv2.imwrite(f'scripts/ArUco_markers/marker_{marker_id}.png', marker_image)
plt.imshow(marker_image, cmap='gray', interpolation='nearest')
plt.axis('off')  # Hide axes
plt.title(f'ArUco Marker {marker_id}')
plt.show()"""



import numpy as np
import cv2
import cv2.aruco as aruco
import os
import argparse
from utils.skeleton_tracker import SkeletonTracker
import pyrealsense2 as rs

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



def correct_rotation_matrix(rotation_matrix):
    rot = rotation_matrix[:3, :3]
    U, _, Vt = np.linalg.svd(rot)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1   # flip last column
        R = np.dot(U, Vt)
    rotation_matrix[:3, :3] = R
    return rotation_matrix




def main():
    # Create pipeline and start config
    align = rs.align(rs.stream.color) # Allinea depth a color
    ctx = rs.context()
    devices = ctx.devices  # Query connected devices
    
    for i, device in enumerate(devices):
        tracker = SkeletonTracker(device.get_info(rs.camera_info.serial_number), 1920, 1080, 30, False)
        mark = MarkerDetector(tracker)

        rotation_matrices = []
        for marker_ID in marker_IDs:
            serial = device.get_info(rs.camera_info.serial_number)
            rotation_matrix = mark.calibration(marker_ID)
            if not rotation_matrix is None:
                rotation_matrices.append(rotation_matrix)

        if len(rotation_matrices) == 0:
            print(f"Calibration failed for device {i} (SN: {serial}). No marker detected.")
            return
        elif len(rotation_matrices) > 0 and len(rotation_matrices) <= len(marker_IDs):
            avg_rotation_matrix = np.mean([rotation_matrix for rotation_matrix in rotation_matrices], axis=0)

            print(f"Media: {avg_rotation_matrix}")
            matrix = correct_rotation_matrix(avg_rotation_matrix)

            print(f"Finl: {matrix}")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_file = os.path.join(script_dir, f"calibration/pose_{serial}.txt")
            write_rotation_matrix_to_file(save_file, matrix)

    print(f"Calibration ended correctly. Marker was detected by all the devices.")



if __name__ == '__main__':
    main()
    

    