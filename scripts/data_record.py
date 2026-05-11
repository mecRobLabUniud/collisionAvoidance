#!/usr/bin/env python3

"""
░█▀▄░█▀█░▀█▀░█▀█░░░█▀▄░█▀▀░█▀▀░█▀█░█▀▄░█▀▄░█▀▀░█▀▄
░█░█░█▀█░░█░░█▀█░░░█▀▄░█▀▀░█░░░█░█░█▀▄░█░█░█▀▀░█▀▄
░▀▀░░▀░▀░░▀░░▀░▀░░░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀▀░░▀▀▀░▀░▀
"""

import sys
import zmq
import time
import numpy as np
import cv2
import os
import json
import struct
import multiprocessing.resource_tracker as rt
from multiprocessing import shared_memory
import threading

TARGET_KEYPOINTS = list(range(17))  # 0..12 pelvis-up
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
EDGES = [(a, b) for (a, b) in COCO_SKELETON if a in TARGET_KEYPOINTS and b in TARGET_KEYPOINTS]

# Parameters
in_port = 6000
out_port = 6000
topic = "SKEL"
running = True
camera = dict(up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.1),
        eye=dict(x=0, y=2, z=0.5))
data = None
pic = None
interfaces = None
skel_len = 17
H, W, C = 480, 848, 3
marker_sz = 8
line_wdt = 5
t0 = time.time()
script_dir = os.path.dirname(os.path.abspath(__file__))
stream_cnt = 0
n_devices = 0
frame_id = 0
paused = False

# ─────────────────────────────────────────────────────────────────────────────
# Frame I/O  (binary .bin + .idx)
#
# .bin layout:  raw H×W×3 uint8 bytes, frames concatenated
# .idx layout:  one 8-byte little-endian int64 per frame = byte offset in .bin
#               → frame N starts at offset stored at position N in the index
#
# Writing is O(1) per frame.
# Reading frame N is O(1): seek to idx[N], read H*W*3 bytes.
# ─────────────────────────────────────────────────────────────────────────────

FRAME_BYTES = H * W * C  # fixed for the whole recording session


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


def count_frames(idx_path: str) -> int:
    """Return the total number of recorded frames in the index."""
    if not os.path.exists(idx_path):
        return 0
    return os.path.getsize(idx_path) // 8


# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────

def record_data(sockets, shms, skeletons_filename, frames_bin, frames_idx):
    global frame_id
    for n in range(int(n_devices)):
        skeleton = sockets[n].recv_string()
        frame = np.ndarray((H, W, C), dtype=np.uint8, buffer=shms[n].buf).copy()

        with open(skeletons_filename[n], "a") as file:
            file.write(skeleton + "\n")

        write_frame(frames_bin[n], frames_idx[n], frame)
        frame_id += 1


# ─────────────────────────────────────────────────────────────────────────────
# Streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_data(socket, shms, skeletons_filename, frames_bin, frames_idx):
    global stream_cnt

    print(f"Streaming frame {stream_cnt}")
    for n in range(n_devices):
        with open(skeletons_filename[n], "r") as file:
            lines = file.readlines()
            if stream_cnt >= len(lines):
                print(f"[stream_data] no more skeleton lines for device {n}")
                return "reset"
            skeleton_line = lines[stream_cnt]

        x, _, msg1, msg2 = skeleton_line.split("; ", 3)
        skeleton = json.loads(msg1)
        confidence = json.loads(msg2)

        payload = (skeleton, confidence)
        message = (f"{topic}_{n}; {n_devices}; "f"{json.dumps(payload[0])}; {json.dumps(payload[1])}")
        socket.send_string(message)

        frame = read_frame(frames_bin[n], frames_idx[n], stream_cnt)
        if frame is None:
            print(f"[stream_data] could not load frame {stream_cnt}")
            return "reset"

        # Write into shared memory
        buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shms[n].buf)
        buf[:] = frame[:]

    stream_cnt += 1


# ─────────────────────────────────────────────────────────────────────────────
# Filename helpers  (now returns .bin + .idx instead of a single .txt)
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
            
    return skeletons_filename, frames_bin, frames_idx


# ─────────────────────────────────────────────────────────────────────────────
# ZeroMQ / shared-memory setup  (unchanged logic, minor cleanup)
# ─────────────────────────────────────────────────────────────────────────────

def setup_recording():
    zctx = zmq.Context.instance()
    probe = zctx.socket(zmq.SUB)
    probe.setsockopt_string(zmq.SUBSCRIBE, topic)
    probe.connect(f"tcp://localhost:{in_port}")
    _, n_dev, _ = probe.recv_string().split("; ", 2)
    probe.close()

    sockets = []
    shms = []
    for n in range(int(n_dev)):
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt_string(zmq.SUBSCRIBE, f"{topic}_{n}")
        sock.connect(f"tcp://localhost:{in_port}")
        sockets.append(sock)

        shm = shared_memory.SharedMemory(name=f"shared_image_{n}")
        rt.unregister(f"/{shm.name}", "shared_memory")
        shms.append(shm)

    return sockets, shms, n_dev


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
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Start input listener in background thread
    input_thread = threading.Thread(target=listen_for_input, daemon=True)
    input_thread.start()

    if arg is None:
        raise ValueError("No argument provided. Use --record or --stream.")

    if arg in ["--record", "-r"]:
        while running:
            res = input("\nRecorded data are still presents in the working directory. Do you want to overwrite them? (y/n)\n")
            if res == "y":
                print("Recording mode enabled. Press Ctrl+C to stop.")
                sockets, shms, n_devices = setup_recording()
                skeletons_filename, frames_bin, frames_idx = init_filenames("r")
                while True:
                    record_data(sockets, shms, skeletons_filename, frames_bin, frames_idx)
                    time.sleep(0.05)
            if res == "n":
                break
            else:
                print("Unknown answer")
                continue

    elif arg in ["--stream", "-s"]:
        print("Streaming mode enabled. Press Ctrl+C to stop.")
        data_dir = os.path.join(script_dir, "data")
        n_devices = sum(1 for f in os.listdir(data_dir) if f.endswith(".bin"))
        skeletons_filename, frames_bin, frames_idx = init_filenames("s")
        socket, shms = setup_streaming(frames_bin[0], frames_idx[0])
        time.sleep(1)
        try:
            while True:
                if not paused:
                    res = stream_data(socket, shms, skeletons_filename, frames_bin, frames_idx)
                    print(n_devices)
                    if res == "reset":
                        stream_cnt = 0
                    time.sleep(0.05)
                else:
                    time.sleep(0.1)                
        finally:
            for shm in shms:
                shm.close()
                shm.unlink()

    else:
        raise ValueError(f"Unknown argument: {arg}")

    quit()


if __name__ == "__main__":
    main()