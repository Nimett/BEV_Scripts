#!/usr/bin/env python

import os
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage


def extract_images(bag_file, output_dir, topic_map):
    timesteps = []
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
                    timesteps.append(timestamp)

                if "/{topic}/right/camera_info" in connection.topic:
                    cam_info_path = Path(output) / "front_cam_info.npy"
                    if not os.path.exists(cam_info_path):
                        msg = bag.deserialize(rawdata, connection.msgtype)
                        cam_info = msg.P.reshape(3, 4)[:3, :3]
                        np.save(cam_info_path, cam_info)

        np.save(f"{output_dir}/front_cam_times.npy", np.array(timesteps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_topic_name", type=str, required=True)
    args = parser.parse_args()

    topic_map = {
        f"{args.img_topic_name}": "front_cam",
    }

    extract_images(args.bag_file, args.output_dir, topic_map)
