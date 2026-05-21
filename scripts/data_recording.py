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
# Frames writing
# ─────────────────────────────────────────────────────────────────────────────
def write_frame(bin_path: str, idx_path: str, frame: np.ndarray):
    """Append one BGR/RGB frame to the binary store and update the index."""
    # Make sure the frame is exactly H×W×C uint8
    assert frame.shape == (H, W, C), f"Unexpected frame shape {frame.shape}"
    raw = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()

    # Current end of .bin = byte offset of the new frame
    offset = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0

    with open(bin_path, "ab") as f:
        f.write(raw)

    with open(idx_path, "ab") as f:
        f.write(struct.pack("<q", offset))   # int64 little-endian


# ─────────────────────────────────────────────────────────────────────────────
# Frames reading
# ─────────────────────────────────────────────────────────────────────────────
def read_frame(bin_path: str, idx_path: str, target_frame_id: int):
    """Return frame at *target_frame_id* as a (H, W, C) uint8 ndarray, or None."""
    idx_size = os.path.getsize(idx_path) if os.path.exists(idx_path) else 0
    n_frames = idx_size // 8

    if target_frame_id >= n_frames:
        print(f"[read_frame] frame {target_frame_id} not found (total: {n_frames})")
        return None

    with open(idx_path, "rb") as f:
        f.seek(target_frame_id * 8)
        offset = struct.unpack("<q", f.read(8))[0]

    with open(bin_path, "rb") as f:
        f.seek(offset)
        raw = f.read(FRAME_BYTES)

    if len(raw) != FRAME_BYTES:
        print(f"[read_frame] short read at offset {offset}: got {len(raw)} bytes")
        return None

    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W, C).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Count frames
# ─────────────────────────────────────────────────────────────────────────────
def count_frames(idx_path: str) -> int:
    """Return the total number of recorded frames in the index."""
    if not os.path.exists(idx_path):
        return 0
    return os.path.getsize(idx_path) // 8


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
        

    # print(f"\rStreaming frame {stream_cnt}", end="")
    # for n in range(n_devices):
    #     with open(skeletons_filename[n], "r") as file:
    #         lines = file.readlines()
    #         if stream_cnt >= len(lines):
    #             print(f"[stream_data] no more skeleton lines for device {n}")
    #             return "reset"
    #         skeleton_line = lines[stream_cnt]
# 
    #     x, _, msg1, msg2 = skeleton_line.split("; ", 3)
    #     skeleton = json.loads(msg1)
    #     confidence = json.loads(msg2)
# 
    #     payload = (skeleton, confidence)
    #     message = (f"{topic}_{n}; {n_devices}; "f"{json.dumps(payload[0])}; {json.dumps(payload[1])}")
    #     socket.send_string(message)
# 
    #     frame = read_frame(frames_bin[n], frames_idx[n], stream_cnt)
    #     if frame is None:
    #         print(f"[stream_data] could not load frame {stream_cnt}")
    #         return "reset"
# 
    #     # Write into shared memory
    #     buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shms[n].buf)
    #     buf[:] = frame[:]

    stream_cnt += 1


# ─────────────────────────────────────────────────────────────────────────────
# Filename helpers
# ─────────────────────────────────────────────────────────────────────────────
def init_filenames(arg):
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    skeletons_filename = [os.path.join(data_dir, f"skeleton_{n}.txt") for n in range(int(n_devices))]
    frames_bin = [os.path.join(data_dir, f"frame_{n}.bin") for n in range(int(n_devices))]
    frames_idx = [os.path.join(data_dir, f"frame_{n}.idx") for n in range(int(n_devices))]

    if arg == "r":
        for n in range(int(n_devices)):
            open(skeletons_filename[n], "w")
            open(frames_bin[n], "w")
            open(frames_idx[n], "w")

    color_writer = VideoRecorder(f"media/color_{self.device}.avi", "XVID", 60, (848, 480), is_color=True)
    depth_writer = VideoRecorder(f"media/depth_{self.device}.avi", "XVID", 60, (848, 480), is_color=True)
            
    return skeletons_filename, frames_bin, frames_idx


# ─────────────────────────────────────────────────────────────────────────────
# ZeroMQ and shared-memory setup fro outgoing data
# ─────────────────────────────────────────────────────────────────────────────
def setup_streaming(frames_bin_0: str, frames_idx_0: str):
    zctx = zmq.Context.instance()
    socket = zctx.socket(zmq.PUB)
    socket.bind(f"tcp://*:{out_port}")

    # Wait until at least one frame is available
    while count_frames(frames_idx_0) == 0:
        time.sleep(0.1)

    frame = read_frame(frames_bin_0, frames_idx_0, 0)
    assert frame is not None, "Could not read first frame from binary store"
    print(f"First frame shape: {frame.shape}")

    print("==============> ", frame.nbytes)

    shms = []
    for i in range(int(n_devices)):
        try:
            shm = shared_memory.SharedMemory(create=True, size=frame.nbytes, name=f"shared_image_{i}")
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=f"shared_image_{i}")
            existing.close()
            existing.unlink()
            shm = shared_memory.SharedMemory(create=True, size=frame.nbytes, name=f"shared_image_{i}")
        shms.append(shm)

    return socket, shms


# ─────────────────────────────────────────────────────────────────────────────
# Entry point 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global n_devices, stream_cnt
    n_devices = 2
    
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # # Start input listener in background thread
    # input_thread = threading.Thread(target=listen_for_input, daemon=True)
    # input_thread.start()

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