#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▀▄░█▀▀░█▀▀░█▀█░█▀▄░█▀▄░▀█▀░█▀█░█▀▀
░█░█░█▀█░░█░░█▀█░░░█▀▄░█▀▀░█░░░█░█░█▀▄░█░█░░█░░█░█░█░█
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀▀░░▀▀▀░▀░▀░▀▀▀
"""

import sys
import zmq
import time
import numpy as np
import cv2
import os
import json
import signal
import struct
import threading
import multiprocessing.resource_tracker as rt
from multiprocessing import shared_memory
from utils.data_transmitter import DataTransmitter
from utils.video_recorder import VideoRecorder

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
in_port = 6000
out_port = 6000
topic = "SKEL"
running = True
skel_len = 17
H, W, C = 480, 848, 3
FRAME_BYTES = H * W * C

stream_cnt = 0
n_devices = 0
frame_id = 0
paused = False
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pause/resume logic
# ─────────────────────────────────────────────────────────────────────────────
def listen_for_input():
    """Listen for keyboard input in a separate thread."""
    global paused
    while True:
        key = input()
        if key.strip().lower() == '' and paused:
            paused = False
            print("\n▶  Loop RESUMED")
        elif key.strip().lower() == '' and not paused:
            paused = True
            print("\n⏸  Loop PAUSED")
 

# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────
def record_data(dtrs, skeleton_data_writers, color_writers):
    global frame_id

    # skeletons = [dtr.receive_skeleton_data()[0] for dtr in dtrs]
    # confidences = [dtr.receive_skeleton_data()[1] for dtr in dtrs]

    for dtr, skeleton_data_writer, color_writer in zip(dtrs, skeleton_data_writers, color_writers):
        skeleton_data_packed = dtr.receive_packed_skeleton_data()

        with open(skeleton_data_writer, "a") as file:
            file.write(skeleton_data_packed + "\n")

        raw_frame = dtr.receive_raw_frames()
        color_writer.write(raw_frame)
        print(raw_frame.shape)

        # write_frame(frames_bin[n], frames_idx[n], frame)
        # frame_id += 1


# ─────────────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────────────
def stream_data(dtss, skeleton_data_readers, color_readers):
    global stream_cnt

    for dts, skeleton_data_reader, color_reader in zip(dtss, skeleton_data_readers, color_readers):
        with open(skeleton_data_reader, "r") as file:
            lines = file.readlines()
            if stream_cnt >= len(lines):
                print(f"[stream_data] no more skeleton lines")
                return "reset"
            skeleton_data_packed = lines[stream_cnt]

            _, skeleton_packed, confidence_packed = skeleton_data_packed.split("; ", 2)
            skeleton = np.array(json.loads(skeleton_packed))
            confidence = np.array(json.loads(confidence_packed))
            dts.send_skeleton_data(skeleton, confidence)

        #with open(color_reader, "r") as file:
        ret, frame = color_reader.read()
        if ret:
            dts.send_frames(frame)

    stream_cnt += 1


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global n_devices, stream_cnt
    n_devices = 2
    
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Start input listener in background thread
    input_thread = threading.Thread(target=listen_for_input, daemon=True)
    input_thread.start()

    # Clear shutdown logic
    def signal_handler(sig, frame):
        global running
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if arg is None:
        raise ValueError("No argument provided. Use --record or --stream.")

    if arg in ["--record", "-r"]:
        while True:
            res = input("\nRecorded data are already present in the working directory. Do you want to overwrite them? (y/n)\n")
            if res == "y":
                print("Recording mode enabled. Press Ctrl+C to stop.")
                dtrs = [DataTransmitter("receiver", n, "SINGLE_CAMERA") for n in range(n_devices)]
                skeleton_data_writers = [os.path.join(data_dir, f"skeleton_data/skeleton_{n}.txt") for n in range(int(n_devices))]
                [open(skeleton_data_writer, "w") for skeleton_data_writer in skeleton_data_writers]
                color_writers = [VideoRecorder(os.path.join(data_dir, f"media/color_{n}.avi"), "XVID", 100, (848, 480), is_color=True) for n in range(n_devices)]
                while running:
                    record_data(dtrs, skeleton_data_writers, color_writers)
                    time.sleep(0.01)
                for dtr, color_writer in zip(dtrs, color_writers):
                    dtr.shutdown()
                    color_writer.release()
            if res == "n":
                break
            else:
                print("Unknown answer")
                continue

    elif arg in ["--stream", "-s"]:
        print("Streaming mode enabled. Press Ctrl+C to stop.")
        dtss = [DataTransmitter("sender", n, "SINGLE_CAMERA") for n in range(n_devices)]
        skeleton_data_readers = [os.path.join(data_dir, f"skeleton_data/skeleton_{n}.txt") for n in range(int(n_devices))]
        color_readers = [cv2.VideoCapture(os.path.join(data_dir, f"media/color_{n}.avi")) for n in range(n_devices)]
        time.sleep(1)
        try:
            while running:
                if not paused:
                    res = stream_data(dtss, skeleton_data_readers, color_readers)
                    if res == "reset":
                        color_readers = [cv2.VideoCapture(os.path.join(data_dir, f"media/color_{n}.avi")) for n in range(n_devices)]
                        stream_cnt = 0
                    time.sleep(0.01)
                else:
                    time.sleep(0.01)                
        finally:
            for dts in dtss:
                dts.shutdown()

    else:
        raise ValueError(f"Unknown argument: {arg}")

    quit()


if __name__ == "__main__":
    main()