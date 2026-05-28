#!/bin/bash

mode=$1
use_gui=$2
n_devices=lsusb | grep 8086 | wc -l

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo $mode
echo $use_gui
echo $n_devices

# trap 'kill 0' INT

camera_stream="python3 scripts/camera_stream.py"
data_recording="python3 scripts/data_recording.py"
data_merging="python3 scripts/data_merging.py"
web_interface="python3 scripts/web_interface.py"


# $camera_stream &
# $data_merging & 
# 
# sleep 1
# $web_interface


if [ "$mode" == "--tracking" ]; then
    echo "Starting tracking procedure..."
    $data_merging & 
    sleep 0.1
    $camera_stream &
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface
        pgrep -f "$web_interface" | xargs kill
    else
        wait
    fi
    pgrep -f "$data_merging" | xargs kill
    pgrep -f "$camera_stream" | xargs kill

elif [ "$mode" == "--recording" ]; then
    echo "Starting recording procedure..."
    $data_recording "-r" &
    sleep 0.1
    $camera_stream &
    wait
    pgrep -f "$camera_stream" | xargs kill
    pgrep -f "$data_recording" | xargs kill

elif [ "$mode" == "--streaming" ]; then
    echo "Starting streaming procedure..."
    $data_recording "-s" &
    $data_merging & 
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface
        pgrep -f "$web_interface" | xargs kill
    else
        wait
    fi
    pgrep -f "$data_recording" | xargs kill
    pgrep -f "$data_merging" | xargs kill
fi

# pgrep -f "$web_interface" | xargs kill
# pgrep -f "$data_merging" | xargs kill
# pgrep -f "$camera_stream" | xargs kill
