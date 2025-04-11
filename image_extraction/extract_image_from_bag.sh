#! /bin/bash

Check if the Docker image ros2_img_ext exists
if ! docker images | grep -q "ros2_img_ext"; then
    echo "Docker image ros2_img_ext not found. Building the image..."
    docker build -t ros2_img_ext .
else
    echo "Docker image ros2_img_ext found."
fi

# docker build -t ros2_img_ext .

bag_file=$1
output_dir=$2
img_topic_name=$3

img_topic_name=${img_topic_name:-"/front_cam/stereo/image_rect_color/compressed"}

cmd="bag_file=$bag_file && \
output_dir=$output_dir && \
img_topic_name=$img_topic_name "

cmd+="export HF_HOME=/tmp  && \
    cd /home/root/humble_ws && \
    source /home/root/.bashrc && \
    exec python3 /home/root/humble_ws/src/img_ext/img_ext/extract_images.py \
    --bag_file $bag_file \
    --output_dir $output_dir"

echo $cmd

exec docker run \
    --gpus 1 \
    -v $bag_file:$bag_file \
    -v $output_dir:$output_dir \
    --user $(id -u):$(id -g) \
    ros2_img_ext \
    /bin/bash -c "$cmd"
