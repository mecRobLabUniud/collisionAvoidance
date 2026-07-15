import cv2
import mediapipe as mp
import numpy as np
import time
import pyrealsense2 as rs

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,      # False for video/stream (uses tracking between frames)
    model_complexity=0,           # 0=lite, 1=full, 2=heavy
    smooth_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

w_camera = 848
h_camera = 480
ctx = rs.context()
devices = ctx.devices
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device(devices[0].get_info(rs.camera_info.serial_number))
cfg.enable_stream(rs.stream.depth, w_camera, h_camera, rs.format.z16, 60)
cfg.enable_stream(rs.stream.color, w_camera, h_camera, rs.format.bgr8, 60)
pipe.start(cfg)

fs = pipe.wait_for_frames()
align = rs.align(rs.stream.color)
fs = align.process(fs)
depth = fs.get_depth_frame()
color = fs.get_color_frame()


annotated = None




def main():
    global annotated

    while True:
        fs = pipe.wait_for_frames()
        color = fs.get_color_frame()

        t0 = time.time()
        color_img = np.asanyarray(color.get_data())
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_img)
        print(f"Inference time: {time.time() - t0:.3f} seconds", end="\r")
        

        # Draw on a copy (draw in BGR since that's what cv2 will save/show)
        annotated = color_img.copy()

        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        

        cv2.imshow("Holistic", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    


if __name__ == "__main__":
    main()
    # Save or display
    cv2.imshow("Holistic", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()