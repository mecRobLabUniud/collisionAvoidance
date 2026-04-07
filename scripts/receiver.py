#!/usr/bin/env python3

from multiprocessing import shared_memory
from PIL import Image
import numpy as np
import cv2
import multiprocessing.resource_tracker as rt
import time

def remove_shm_from_resource_tracker(name):
    rt.unregister(f"/{name}", "shared_memory")

# Must know shape/dtype in advance (or pass via a side channel)
H, W, C = 480, 848, 3  # adjust to match your image
dtype = np.uint8



# Convert back to PIL image and use it
while True:
    t0 = time.time()
    # Attach to existing shared memory block
    shm = shared_memory.SharedMemory(name="shared_image")
    remove_shm_from_resource_tracker(shm.name) 

    # Read image data from shared memory
    arr = np.ndarray((H, W, C), dtype=dtype, buffer=shm.buf)

    # Make a copy before detaching (important!)
    img_array = arr.copy()

    shm.close()  # Detach (do NOT unlink here — sender owns it)

    t1 = time.time()
    print(f"\rElapsed time: {t1-t0} s", end=" ")

    cv2.imshow(f"YOLO Skeleton Realtime Camera", img_array)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break