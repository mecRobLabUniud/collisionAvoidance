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

# parameters
marker_ID = 42
dim = 0.05
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)



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



def inversePerspective(rot, pos):
    R, _ = cv2.Rodrigues(rot)
    R = np.matrix(R).T
    invpos = np.dot(-R, np.matrix(pos))
    invrot, _ = cv2.Rodrigues(R)
    return invrot, invpos



def relativePosition(rot1, pos1):
    rot1, pos1 = rot1.reshape((3, 1)), pos1.reshape(
        (3, 1))
    # rot2, pos2 = rot2.reshape((3, 1)), pos2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invrot, invpos = inversePerspective(rot2, pos2)

    orgrot, orgpos = inversePerspective(invrot, invpos)
    # print("rot: ", rot2, "pos: ", pos2, "\n and \n", orgrot, orgpos)

    info = cv2.composeRT(rot1, pos1, invrot, invpos)
    composedrot, composedpos = info[0], info[1]

    composedrot = composedrot.reshape((3, 1))
    composedpos = composedpos.reshape((3, 1))
    return composedrot, composedpos



def recognition(tracker, align, matrix_coefficients, distortion_coefficients):
    pointCircle = (0, 0)
    markerposList = []
    markerrotList = []
    composedrot, composedpos = None, None
    while True:
        frame = tracker.get_color_frame()
        
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()  # new style
        detector = aruco.ArucoDetector(dictionary, parameters)

        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = detector.detectMarkers(gray)

        if np.all(ids is not None):  # If there are markers found by detector
            del markerposList[:]
            del markerrotList[:]
            zipped = zip(ids, corners)
            ids, corners = zip(*(sorted(zipped)))
            axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rot and pos---different from camera coefficients
                rot, pos, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
                if ids[i] == marker_ID:
                    rot = rot
                    pos = pos
                    isFirstMarkerCalibrated = True
                    firstMarkerCorners = corners[i]

                # print(markerPoints)
                (rot - pos).any()  # get rid of that nasty numpy value array error
                markerrotList.append(rot)
                markerposList.append(pos)

                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

            imgpts, jac = cv2.projectPoints(axis, rot, pos, matrix_coefficients,
                                            distortion_coefficients)

            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rot, pos, length=0.1)
            relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
            cv2.circle(frame, relativePoint, 2, (255, 255, 0))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Calibration
            if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                rot, pos = rot.reshape((3, 1)), pos.reshape((3, 1))
                secondrot, secondpos = secondrot.reshape((3, 1)), secondpos.reshape((3, 1))

                composedrot, composedpos = relativePosition(rot, pos, secondrot, secondpos)

        # Convert Rodrigues rot to 3x3 rotation matrix
        R_mat, _ = cv2.Rodrigues(rot)
        
        # Build 4x4 pose matrix [R | t; 0 0 0 1]
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = R_mat  # Rotation part
        pose_matrix[:3, 3] = pos.flatten()  # Translation part

    cv2.destroyAllWindows()

    return pose_matrix



def calibration(frame, matrix_coefficients, distortion_coefficients):   
    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()  # new style
    detector = aruco.ArucoDetector(dictionary, parameters)

    # lists of ids and the corners beloning to each id
    corners, ids, _ = detector.detectMarkers(gray)

    pose_matrix = None
    if np.all(ids is not None):
        zipped = zip(ids, corners)
        ids, corners = zip(*(sorted(zipped)))
        axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
        # Estimate pose of each marker
        for i in range(len(ids)):
            if ids[i] == marker_ID:
                rot, pos, _ = aruco.estimatePoseSingleMarkers(corners[i], dim, matrix_coefficients, distortion_coefficients)

                # Build 4x4 pose matrix [R | t; 0 0 0 1]
                R_mat, _ = cv2.Rodrigues(rot)
                pose_matrix = np.eye(4, dtype=np.float32)
                pose_matrix[:3, :3] = R_mat  # Rotation part
                pose_matrix[:3, 3] = pos.flatten()  # Translation part

                pose_matrix = np.linalg.inv(pose_matrix)

                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                imgpts, jac = cv2.projectPoints(axis, rot, pos, matrix_coefficients,
                                                distortion_coefficients)

                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rot, pos, length=0.1)
                relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                cv2.circle(frame, relativePoint, 2, (255, 255, 0))

                # Display the resulting frame
                cv2.imshow('frame', frame)
                # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
                key = cv2.waitKey(1000) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('c'):  # Calibration
                    if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                        rot, pos = rot.reshape((3, 1)), pos.reshape((3, 1))
                        secondrot, secondpos = secondrot.reshape((3, 1)), secondpos.reshape((3, 1))

                        composedrot, composedpos = relativePosition(rot, pos, secondrot, secondpos)

    return pose_matrix



def write_pose_matrix_to_file(file, mat):
    with open(file, 'w') as file:
        for i in range(4):
            file.write('\t'.join(map(str, mat[i, :])) + '\n')



def main():
    # Create pipeline and start config
    align = rs.align(rs.stream.color) # Allinea depth a color
    ctx = rs.context()
    devices = ctx.devices  # Query connected devices
    for i, device in enumerate(devices):
        tracker = SkeletonTracker(device.get_info(rs.camera_info.serial_number))

        (mtx, dist) = tracker.get_intrinsics()
        serial = device.get_info(rs.camera_info.serial_number)

        # while True:
        #     _, frame = tracker.get_color_frame(align)
        #     recognition(mtx, dist)
        
        frame = tracker.get_color_frame()
        pose_matrix = calibration(frame, mtx, dist)
        rot_matrix = np.array([[1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])
        # pose_matrix = np.dot(rot_matrix, pose_matrix)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_file = os.path.join(script_dir, f"calibration/pose_{serial}.txt")
        write_pose_matrix_to_file(save_file, pose_matrix)

        # recognition(tracker, align, mtx, dist)



if __name__ == '__main__':
    main()
    

    