#!/bin/bash

mode=$1
use_gui=$2

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo $mode
echo $use_gui

trap 'kill 0' INT

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
    $camera_stream &
    $data_merging & 
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface
    else
        wait
    fi
elif [ "$mode" == "--recording" ]; then
    echo "Starting recording procedure..."
elif [ "$mode" == "--streaming" ]; then
    echo "Starting streaming procedure..."
    $data_recording "-s" &
    $data_merging & 
    if [ "$use_gui" == "--gui" ]; then
        sleep 1
        $web_interface
    else
        wait
    fi
fi

# pgrep -f "$web_interface" | xargs kill
# pgrep -f "$data_merging" | xargs kill
# pgrep -f "$camera_stream" | xargs kill
