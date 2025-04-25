#!/usr/bin/env python

from argparse import ArgumentParser
import rospy
import tf
import numpy as np
import pickle


def tf_listener(time_transform_file):
    """Listens to the /odom -> /front_cam_left_camera_optical_frame transform at specific timestamps
    and saves the corresponding translations and rotations into the time_transform_file.
    """

    rospy.init_node("tf_listener_node", anonymous=True)

    print("tf_listener_node is starting")

    tf_listener = tf.TransformListener()

    rate = rospy.Rate(10.0)

    with open(time_transform_file, "rb") as f:
        time_tra = pickle.load(f)

    timesteps_to_fill = [t for t in time_tra if len(time_tra[t]) == 0]
    print(f"There are {len(timesteps_to_fill)} timestamps to fill")

    count = 0

    while not rospy.is_shutdown() and len(timesteps_to_fill) != 0:
        try:
            ns = timesteps_to_fill[0]
            secs = ns // 1000000000
            nsecs = ns % 1000000000

            query_time = rospy.Time(secs, nsecs)
            real_time = rospy.Time().now()

            (trans, rot) = tf_listener.lookupTransform(
                "/odom", "/front_cam_left_camera_optical_frame", query_time
            )
            time_tra[ns] = [*trans, *rot]
            timesteps_to_fill.remove(ns)

            start_time = rospy.Time.now()
            count += 1

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            if count > 0:
                time_diff = (real_time - start_time).to_sec()

                if time_diff > 120:
                    with open(time_transform_file, "wb") as f:
                        pickle.dump(time_tra, f)
                    print(f"Saved transform tree to {time_transform_file}")

                    timesteps_to_fill.remove(ns)
            continue

        rate.sleep()

    with open(time_transform_file, "wb") as f:
        pickle.dump(time_tra, f)

    print(f"Saved transform tree to {time_transform_file} and the script finished")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--time_transform_file",
        type=str,
        required=True,
        help="Path to npy file that has timestamps of images",
    )
    args = parser.parse_args()
    tf_listener(args.time_transform_file)
