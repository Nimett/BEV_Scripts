# BEV_Scripts

## Extract stereo images from a rosbag
```
cd image_extraction
bash extract_image_from_bag.sh <path/to/bagfile> <path/to/output_directory>
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

### Extract Images from Rosbag