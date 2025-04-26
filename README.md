# ros-bev-generator

This repository is designed to generate semantic Bird’s-Eye View (BEV) maps using stereo images and TF Tree from ROS bag files. These bag files include:

	- Stereo camera images
	- The robot’s position (odometry to camera transform), computed using ROS TF

Generated semantic BEV maps can be used for training or evaluating perception models that rely on accurate spatial and visual data in BEV format.

Follow the steps below in order to generate BEV maps.

## Extract stereo images from a rosbag
```bash
cd image_extraction
bash extract_image_from_bag.sh <path/to/bagfile> <path/to/parent_output_directory>
```

## Extract Odom to Camera Transform from a rosbag

### Start roscore
```bash
roscore
```
### In another terminal, run the script
```
source /opt/ros/<your_ros_distribution>/setup.bash
cd transform_tree_extraction
bash tf_listener.sh <path/to/your/bag/file> <path/to/parent_output_directory>
```

## Generate disparity and depth maps from stereo image pairs using OpenCV's StereoSGBM
```bash
pip install numpy opencv-python tqdm pyyaml pillow
```
```bash
cd depth_generation
python stereo_matching.py \
    --parent_output_dir <path/to/parent_output_directory> \
    --bag_file_name <name/of/the/bag/file>
```

## Generate prompt-based segmentations using Grounded-SAM for all images in a folder

### Clone the Repository 
```bash
git clone https://github.com/Nimett/Grounded-Segment-Anything.git
```

### Run Batch Segmentation
```bash
bash run_batch_segmentation.sh 
    <path/to/parent_output_directory> 
    <name/of/the/bag/file>
    [image/extension]
    [segmentation/classes]
```
Arguments:

	- [image_extension] – (Optional) Image file extension. Default: png
	- [segmentation_classes] – (Optional) Comma-seperated segmentation classes. Default: "High-standing platforms, Ground, Humans"

## Generate BEV maps

```bash
python generate_bev_maps.py --seg_class_file <path/to/seg_classes.yaml> --parent_output_dir <path/to/parent_output_directory> --bag_file_name <bag_file_name>