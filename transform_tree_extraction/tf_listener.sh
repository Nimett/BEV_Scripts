#!/bin/bash

IMAGE_TIME_FILE=$1
ROS_BAG_FILE=$2


if [ -z "$IMAGE_TIME_FILE" ] || [ -z "$ROS_BAG_FILE" ]; then
    echo "Usage: $0 <image_time_file> <rosbag_file>"
    exit 1
fi

# Run tf_listener.py with the image_time_file parameter
python tf_listener.py --image_time_file=$IMAGE_TIME_FILE &

sleep 1

rosbag play $ROS_BAG_FILE &

# Wait for both background processes to finish
wait