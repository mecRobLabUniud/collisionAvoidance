#!/bin/bash

mode=$1
use_gui=$2

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo $mode

# trap 'kill 0' INT

camera_stream="python3 scripts/camera_stream.py"
data_recording="python3 scripts/data_recording.py"
data_merging="python3 scripts/data_merging.py"
web_interface="python3 scripts/web_interface.py"

$camera_stream &
sleep 2
$data_merging & 
sleep 2
$web_interface 
# wait

pgrep -f "$web_interface" | xargs kill
pgrep -f "$data_merging" | xargs kill
pgrep -f "$camera_stream" | xargs kill

# if [ "$mode" == "--tracking" ]; then
#     echo "Starting tracking procedure..."
#     python3 scripts/camera_stream.py &
#     python3 scripts/data_merging.py &
#     if [$use_gui == "true"]; then
#         python3 scripts/web_interface.py
#     fi
# elif [ "$mode" == "--recording" ]; then
#     echo "Starting recording procedure..."
# elif [ "$mode" == "--streaming" ]; then
#     echo "Starting streaming procedure..."
# fi
