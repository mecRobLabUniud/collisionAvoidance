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
rula_evaluation="$dir/build/rula_evaluation"

if [ "$mode" == "--track" ]; then
    echo "Starting tracking procedure..."
    $data_merging "$n_devices" &
    sleep 0.1
    $camera_stream &
    $rula_evaluation &
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface "$n_devices"
    else
        wait
    fi

elif [ "$mode" == "--record" ]; then
    echo "Starting recording procedure..."
    $data_recording "$n_devices" "-r" &
    sleep 0.1
    $camera_stream &
    wait

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
    $rula_evaluation &
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface "$n_devices"
    else
        wait
    fi

elif [ "$mode" == "--calibrate" ]; then
    $calibration
fi

