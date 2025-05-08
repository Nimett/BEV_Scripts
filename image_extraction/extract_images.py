#!/usr/bin/env python

import os
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np
import pickle
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage


def extract_images(bag_file, output_dir, cam_info_dir, topic_map):
    print("Image Extraction Started")
    timesteps = {}
    output = output_dir

    with AnyReader([Path(bag_file)]) as bag:
        for topic in topic_map:
            cam_left_output_dir = os.path.join(output, topic_map[topic], "left", "")
            cam_right_output_dir = os.path.join(output, topic_map[topic], "right", "")

            if not os.path.isdir(os.path.dirname(cam_left_output_dir)):
                os.makedirs(os.path.dirname(cam_left_output_dir))
            if not os.path.isdir(os.path.dirname(cam_right_output_dir)):
                os.makedirs(os.path.dirname(cam_right_output_dir))

            for connection, timestamp, rawdata in bag.messages():
                if (
                    connection.topic == topic
                    and connection.msgtype == "sensor_msgs/msg/CompressedImage"
                ):
                    msg = bag.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, "bgr8")
                    h, w, _ = img.shape
                    left_img = img[:, : w // 2, :]
                    right_img = img[:, w // 2 :, :]
                    cv2.imwrite(
                        os.path.join(cam_left_output_dir, str(timestamp) + ".png"),
                        left_img,
                    )
                    cv2.imwrite(
                        os.path.join(cam_right_output_dir, str(timestamp) + ".png"),
                        right_img,
                    )
                    timesteps[timestamp] = []

                if f"/{topic_map[topic]}/right/camera_info" in connection.topic:
                    cam_info_path = Path(cam_info_dir) / "right_cam_info.npy"
                    if not os.path.exists(cam_info_path):
                        msg = bag.deserialize(rawdata, connection.msgtype)
                        cam_info = msg.P.reshape(3, 4)
                        np.save(cam_info_path, cam_info)
                if f"/{topic_map[topic]}/left/camera_info" in connection.topic:
                    cam_info_path = Path(cam_info_dir) / "front_cam_info.npy"
                    if not os.path.exists(cam_info_path):
                        msg = bag.deserialize(rawdata, connection.msgtype)
                        cam_info = msg.P.reshape(3, 4)
                        np.save(cam_info_path, cam_info)

        with open(f"{output_dir}/time_transform_file.pkl", "wb") as f:
            pickle.dump(timesteps, f)

    print("Image Extraction Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_file", type=str, required=True)
    parser.add_argument("--parent_dir", type=str, required=True)
    parser.add_argument("--img_topic_name", type=str, required=True)
    args = parser.parse_args()

    cam_name = args.img_topic_name.split("/")[1]
    topic_map = {
        f"{args.img_topic_name}": cam_name,
    }

    bag_name = Path(args.bag_file.strip()).stem
    output_dir = Path(args.parent_dir)/bag_name
    cam_info_dir = Path(args.parent_dir)

    extract_images(args.bag_file, output_dir, cam_info_dir, topic_map)
