import cv2
import numpy as np
import time
import pyrealsense2 as rs
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

current_dir = os.path.dirname(os.path.abspath(__file__))


class MediapipeTracker():
    def __init__(self): 
        self.start_time = time.time()
        self.base_options = python.BaseOptions(
            model_asset_path=f'{current_dir}/../models/pose_landmarker_full.task',
            delegate=python.BaseOptions.Delegate.GPU,  # falls back to CPU if GPU unavailable
        )
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.VIDEO,   # VIDEO mode uses timestamps for tracking continuity
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)


    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for pose_landmarks in pose_landmarks_list:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


    def inference_pose_landmarker(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        frame_timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        return result


def main():
    # --- RealSense setup ---
    w_camera, h_camera = 848, 480
    ctx = rs.context()
    devices = ctx.devices
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(devices[0].get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.color, w_camera, h_camera, rs.format.bgr8, 60)
    pipe.start(cfg)

    while True:
        t0 = time.time()

        fs = pipe.wait_for_frames()
        color = fs.get_color_frame()
        if not color:
            continue

        color_img = np.asanyarray(color.get_data())
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        mp = MediapipeTracker()

        t1 = time.time()
        result = mp.inference_pose_landmarker(rgb_img)
        t2 = time.time()

        annotated = mp.draw_landmarks_on_image(rgb_img, result)

        total = time.time() - t0
        print(f"inference: {(t2-t1)*1000:.1f}ms | total: {total*1000:.1f}ms | "
              f"FPS: {1/total:.1f}", end="\r")

        cv2.imshow("Pose (Tasks API)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()