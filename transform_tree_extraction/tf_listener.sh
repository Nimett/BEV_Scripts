#!/bin/bash

rosparam set use_sim_time true

sleep 1

ROS_BAG_FILE=$1
PARENT_FOLDER=$2

if [ -z "$PARENT_FOLDER" ] || [ -z "$ROS_BAG_FILE" ]; then
    echo "Usage: $0 <parent_folder> <rosbag_file>"
    exit 1
fi

BAG_NAME=$(basename "$ROS_BAG_FILE" .bag)
TIME_TRANSFORM_FILE="${PARENT_FOLDER}/${BAG_NAME}/time_transform_file.pkl"

while true; do
    python tf_listener.py --time_transform_file="$TIME_TRANSFORM_FILE" &

    sleep 1

    rosbag play "$ROS_BAG_FILE" --clock &
    ROSBAG_PID=$!

    # Wait for rosbag to finish
    wait $ROSBAG_PID

    # Check the pickle contents
    python check_pickle.py "$TIME_TRANSFORM_FILE"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "All data present. Exiting loop."
        break
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "Detected missing data. Re-running the loop..."
    else
        echo "Error during check. Exiting."
        exit 1
    fi
done