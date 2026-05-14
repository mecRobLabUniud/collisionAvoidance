#!/usr/bin/env python3

from utils.data_transmitter import DataTransmitter


dt = DataTransmitter("sender", 0, 0, "")
dt.send_frames(None)