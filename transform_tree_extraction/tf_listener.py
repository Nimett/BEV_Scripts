#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
import rospy
import tf
import numpy as np


def tf_listener(image_time_file):
    """Listens to the /odom -> /front_cam_left_camera_optical_frame transform at specific timestamps
    and saves the corresponding translations and rotations to a file.
    """

    rospy.init_node("tf_listener_node", anonymous=True)
   
    print("tf_listener_node")
   
    tf_listener = tf.TransformListener()

    rate = rospy.Rate(1.0)
    output_folder = Path(image_time_file).parent

    times = np.load(image_time_file)
    count = 0
    tra_arr = []

    while not rospy.is_shutdown() and count < len(times):
        try:
            ns = times[count]
            secs = ns // 1000000000
            nsecs = ns % 1000000000
            curr_time = rospy.Time(secs, nsecs)
            (trans, rot) = tf_listener.lookupTransform(
                "/odom", "/front_cam_left_camera_optical_frame", curr_time
            )
            tra_arr.append([*trans, *rot])
            count += 1
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            continue
        rate.sleep()

    np.save(f"{output_folder}/odom_to_optical_frame.npy", np.array(tra_arr))
    print(f"Saved transform tree to {output_folder}/odom_to_optical_frame.npy")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_time_file",
        type=str,
        required=True,
        help="Path to npy file that has timestamps of images",
    )
    args = parser.parse_args()
    tf_listener(args.image_time_file)
