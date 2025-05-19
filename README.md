# ros-bev-generator

This repository provides a pipeline to generate semantic Bird’s-Eye View (BEV) maps from ROS bag files. These bag files include:

- Stereo camera images
- The robot’s position (odometry to camera transform), computed using ROS TF
- Left and Right Camera Info

Generated semantic BEV maps can be used for training or evaluating perception models that rely on accurate spatial and visual data in BEV format.

## Key Features
- **Stereo Image Extraction**: Extract stereo images from ROS bag files.
- **TF Tree Extraction**: Extract odometry-to-camera transforms using ROS TF.
- **Depth Map Generation**: Compute disparity and depth maps using OpenCV's StereoSGBM.
- **Semantic Segmentation**: Perform prompt-based segmentation using Grounded-SAM.
- **BEV Map Generation**: Generate semantic BEV maps from processed data.

Bag files are assumed to have following topics:
- **/front_cam/stereo/image_rect_color/compressed**

    Rectified and compressed stereo image stream from the front camera.

- **/front_cam/right/camera_info**

    Camera info for the right front stereo camera.

- **/front_cam/left/camera_info**

    Camera info for the left front stereo camera.

- **/tf**

    Must include the transform from /odom to /front_cam_left_camera_optical_frame

## This project was developed and tested with the following environment:
- Operating System: Ubuntu 20.04
- ROS Version: Noetic
- Python Version: 3.10.14

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Nimett/ros-bev-generator.git
   cd ros-bev-generator
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow Overview
Follow these steps to generate BEV maps:

### 1. Extract Stereo Images
Extract stereo images from a ROS bag file:
```bash
cd image_extraction
bash extract_image_from_bag.sh <path/to/bagfile> <path/to/parent_output_directory>
```
### 2. Extract Odometry-to-Camera Transform
Start the ROS core in one terminal:
```bash
roscore
```
In another terminal, run the transform extraction script:
```bash
source /opt/ros/<your_ros_distribution>/setup.bash
cd transform_tree_extraction
bash tf_listener.sh <path/to/your/bag/file> <path/to/parent_output_directory>
```

### 3. Generate Depth Maps from stereo image pairs using OpenCV's StereoSGBM
Run the stereo matching script:
```bash
cd depth_generation
python stereo_matching.py \
    --parent_output_dir <path/to/parent_output_directory> \
    --bag_file_name <name/of/the/bag/file>
```
Note: Provide the bag file name without the .bag extension for the --bag_file_name argument.

### 4. Generate Prompt-based Semantic Segmentation using Grounded-SAM for All Images in a Folder
Clone the [Grounded-Segment-Anything](https://github.com/Nimett/Grounded-Segment-Anything) repository:
```bash
git clone https://github.com/Nimett/Grounded-Segment-Anything.git
```

Run batch segmentation:
```bash
bash run_batch_segmentation.sh 
    <path/to/parent_output_directory> 
    <name/of/the/bag/file>
    [image/extension]
    [segmentation/classes]
```
Note: Provide the bag file name without the .bag extension.

**Arguments**:
- `[image_extension]` (Optional): Image file extension (default: `png`).
- `[segmentation_classes]` (Optional): Comma-separated segmentation classes (default: `"High-standing platforms,Ground,Humans"`).

### 5. Generate BEV Maps
Run the BEV map generation script:
```bash
python generate_bev_maps.py \
    --parent_output_dir <path/to/parent_output_directory> \
    --bag_file_name <bag_file_name> \
    [--seg_class_file <path/to/seg_classes.yaml>]
```
Note: Provide the bag file name without the .bag extension.

**Arguments**:
- `[--seg_class_file]` (Optional): Path to the segmentation class config file (default: semantic_bev_generation/seg_classes.yaml).