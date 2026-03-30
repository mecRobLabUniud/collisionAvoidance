#!/usr/bin/env python3

import zmq
import json

endpoint = "tcp://localhost:5556"
topic = "SKEL"

zctx = zmq.Context.instance()
socket = zctx.socket(zmq.SUB)
socket.connect(endpoint)

# Subscribe to "news" topic (prefix match)
socket.setsockopt_string(zmq.SUBSCRIBE, "SKEL")

while True:
    topic, message = socket.recv_string().split(" ", 1)
    array = json.loads(message)
    print(f"Received: {array}")
    print(f"Type: {type(array)}")
    