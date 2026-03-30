#!/usr/bin/env python3

import zmq
import time
import numpy as np
import struct
import json

endpoint = "tcp://*:5556"
topic = "SKEL"

MAGIC = b"SKEL" 
VERSION = 1
HDR_FMT = "<4sHHQ"     # magic, version, n_caps, t_mono_ns )
# REC_FMT = "<8f"        # x1 y1 z1 x2 y2 z2 radius conf
REC_FMT = "<3f"
MAX_CAPS = 32

zctx = zmq.Context.instance()
socket = zctx.socket(zmq.PUB)
socket.bind(endpoint)

# Give subscribers time to connect
time.sleep(1)

prova = np.array([[    0.26491,    -0.27416,       2.908],
                              [    0.27546,    -0.30485,      2.9586],
                              [    0.20845,    -0.28864,      2.7732],
                              [    0.10197,    -0.25574,      2.7239],
                              [    0.13484,    -0.13318,      2.6583],
                              [    0.12748,    -0.09594,      2.5547],
                              [  -0.032616,   0.0012576,     0.28827],
                              [     0.2746,    0.095331,      2.4282],
                              [   -0.10029,    -0.02732,     0.19259],
                              [    0.39306,   -0.017061,      2.4974],
                              [    0.18929,     0.27972,      2.6906],
                              [    0.17451,     0.30773,       2.665]])
            
# prova = np.ndarray(prova)

t_mono_ns = time.monotonic_ns()
# Header: Magic, Versione, Numero Capsule, Timestamp
# struct.pack(): converte i dati in una stringa di byte secondo il formato specificato
header = struct.pack(HDR_FMT, MAGIC, VERSION, len(prova), t_mono_ns)
# Payload: Lista di capsule (*rec serve a spacchettare la tupla della capsula in singoli argomenti - grazie all'asterisco -)
#payload = b"".join((*pnt) for pnt in prova)
# Invio messaggio completo (header + payload) (singolo messaggio atomico)

payload = json.dumps(prova.tolist())

while True:
    message = f"{topic} {payload}"
    socket.send_string(message)
    print(f"Published: {message}")
    time.sleep(1)