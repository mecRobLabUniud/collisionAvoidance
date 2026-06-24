#!/bin/bash

mode=$1
use_gui=$2
test=$3
n_devices=$(lsusb | grep 8086 | wc -l)

trap 'kill 0' INT

dir="$(dirname $0)"

camera_stream="python3 $dir/scripts/camera_stream.py"
data_recording="python3 $dir/scripts/data_recording.py"
data_merging="python3 $dir/scripts/data_merging.py"
web_interface="python3 $dir/scripts/web_interface.py"
calibration="python3 $dir/scripts/calibration.py"

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

    if [ -z "$test" ]; then
        echo "No test number provided. Please provide a test number."
        echo -n "Enter the value of the test to stream: "
        read n_test
    else
        n_test=$test
    fi
    n_devices=$(ls $dir/scripts/data/skeleton_data/test"$n_test"/skeleton* | wc -l)

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

