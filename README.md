# BEV_Scripts

## Generate disparity and depth maps from stereo image pairs using OpenCV's StereoSGBM

**Location:** `stereo_matching.py`

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