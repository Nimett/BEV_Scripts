# BEV_Scripts

This repository is designed to generate ground truth (GT) Bird’s-Eye View (BEV) maps using ROS bag files. These bag files include:

	- Stereo camera images, extracted from the original dataset
	- The robot’s position (odometry to camera transform), computed using ROS TF

The processed bag files can be used for training or evaluating perception models that rely on accurate spatial and visual data in BEV format.

Follow the steps below in order to generate BEV maps.

## Extract stereo images from a rosbag
```
cd image_extraction
bash extract_image_from_bag.sh <path/to/bagfile> <path/to/parent_output_directory>
```

## Extract Odom to Camera Transform from a rosbag

### Start roscore
```
roscore
```
### In another terminal, run the script
```
source /opt/ros/<your_ros_distribution>/setup.bash
cd transform_tree_extraction
bash tf_listener.sh <path/to/your/bag/file> <path/to/parent_output_directory>
```

## Generate disparity and depth maps from stereo image pairs using OpenCV's StereoSGBM

**Location:** `stereo_matching.py`
```bash
python stereo_matching.py \
    --left_img_path path/to/left_image.png \
    --right_img_path path/to/right_image.png \
    --output_dir path/to/output_directory \
    --cam_info_file
```

## Generate prompt-based segmentations using Grounded-SAM for all images in a folder

### Clone the Repository 
```bash
git clone https://github.com/Nimett/Grounded-Segment-Anything.git
```

### Run Batch Segmentation
```bash
bash run_batch_segmentation.sh <image/folder> <output/segmentation/folder>
```

- <image/folder>: Path to the folder containing images.
- <output/segmentation/folder>: Path where the segmented output will be saved.