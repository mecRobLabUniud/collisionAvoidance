import cv2

video_filename = ~/Desktop/collision_avoidance


def class VideoRecorder():
    def __init__(self):
        self.cap = cv2.VideoCapture("video.mp4")  # or use 0 for webcam
        self.writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 70, (W, H))

"""while cap.isOpened():
    ret, frame = cap.read()  # ret = True if frame was read successfully
    
    if not ret:
        break  # End of video
    
    # `frame` is a NumPy array (H, W, 3) in BGR format
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(25) & 0xFF == ord("q"):  # Press Q to quit
        break

cap.release()
cv2.destroyAllWindows()"""





# # Inizializzazione VideoWriter
    # video_writer = None
    # if save_video:
    #     