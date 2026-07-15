import cv2
import numpy as np
import time
import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

# --- Setup HandLandmarker (Tasks API) ---
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task',
    delegate=python.BaseOptions.Delegate.GPU,  # falls back to CPU if GPU unavailable
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# --- RealSense setup ---
w_camera, h_camera = 848, 480
ctx = rs.context()
devices = ctx.devices
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device(devices[0].get_info(rs.camera_info.serial_number))
cfg.enable_stream(rs.stream.color, w_camera, h_camera, rs.format.bgr8, 60)
pipe.start(cfg)


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Label Left/Right above the wrist
        h, w, _ = annotated_image.shape
        wrist = hand_landmarks[0]
        x_px, y_px = int(wrist.x * w), int(wrist.y * h) - 20
        label = handedness_list[idx][0].category_name
        cv2.putText(annotated_image, label, (x_px, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_image


def main():
    start_time = time.time()

    while True:
        t0 = time.time()

        fs = pipe.wait_for_frames()
        color = fs.get_color_frame()
        if not color:
            continue

        color_img = np.asanyarray(color.get_data())
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        frame_timestamp_ms = int((time.time() - start_time) * 1000)

        t1 = time.time()
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        t2 = time.time()

        annotated = draw_hand_landmarks_on_image(rgb_img, result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        total = time.time() - t0
        print(f"inference: {(t2-t1)*1000:.1f}ms | total: {total*1000:.1f}ms | "
              f"FPS: {1/total:.1f}", end="\r")

        cv2.imshow("Hand Landmarks (Tasks API)", annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()