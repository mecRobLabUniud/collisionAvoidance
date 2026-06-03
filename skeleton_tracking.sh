#!/bin/bash

mode=$1
use_gui=$2
n_devices=$(lsusb | grep 8086 | wc -l)

trap 'kill 0' INT

camera_stream="python3 scripts/camera_stream.py"
data_recording="python3 scripts/data_recording.py"
data_merging="python3 scripts/data_merging.py"
web_interface="python3 scripts/web_interface.py"
calibration="python3 scripts/calibration.py"

if [ "$mode" == "--track" ]; then
    echo "Starting tracking procedure..."
    $data_merging "$n_devices" &
    sleep 0.1
    $camera_stream &
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface "$n_devices"
        # pgrep -f "$web_interface" | xargs kill
    else
        wait
    fi
    # pgrep -f "$data_merging" | xargs kill
    # pgrep -f "$camera_stream" | xargs kill

elif [ "$mode" == "--record" ]; then
    echo "Starting recording procedure..."
    $data_recording "$n_devices" "-r" &
    sleep 0.1
    $camera_stream &
    wait
    # pgrep -f "$camera_stream" | xargs kill
    # pgrep -f "$data_recording" | xargs kill

elif [ "$mode" == "--stream" ]; then
    echo -n "Enter the value of the test to stream: "
    read n_test

    n_devices=$(ls scripts/data/skeleton_data/test"$n_test"/skeleton* | wc -l)

    echo "Starting streaming procedure..."
    $data_recording "$n_devices" "-s" "$n_test" &
    $data_merging "$n_devices" &
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface "$n_devices"
        # pgrep -f "$web_interface" | xargs kill
    else
        wait
    fi
    # pgrep -f "$data_recording" | xargs kill
    # pgrep -f "$data_merging" | xargs kill

elif [ "$mode" == "--calibrate" ]; then
    $calibration
fi

