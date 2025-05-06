from argparse import ArgumentParser
from pathlib import Path
import PIL.Image as Image
import cv2
import numpy as np
from tqdm import tqdm


def load_image_list(left_img_dir, right_img_dir):
    """Loads left/right stereo image paths"""
    left_imgs = {li.stem: li for li in sorted(Path(left_img_dir).glob("*.png"))}
    right_imgs = {ri.stem: ri for ri in sorted(Path(right_img_dir).glob("*.png"))}
    return left_imgs, right_imgs


def create_stereo_matcher(min_disp=0, num_disp=96, block_size=7):
    """Create and configure a StereoSGBM matcher."""
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def process_image(
    img_name,
    left_path,
    right_path,
    disp_save_dir,
    depth_save_dir,
    focal_length,
    baseline,
    stereo_matcher,
):
    """Processes a stereo image pair to compute, inpaint, and save disparity and depth maps"""
    left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    disparity = stereo_matcher.compute(left_img, right_img).astype(np.float32) / 16.0
    invalid_disp_mask = disparity <= 0

    # Save original disparity map as npy and image file
    np.save(disp_save_dir / f"{img_name}_orig.npy", disparity)
    disp_8bit = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    disp_orig_coloured = cv2.applyColorMap(disp_8bit, cv2.COLORMAP_JET)
    disp_orig_coloured[invalid_disp_mask] = [0, 0, 0]
    Image.fromarray(cv2.cvtColor(disp_orig_coloured, cv2.COLOR_BGR2RGB)).save(
        disp_save_dir / f"{img_name}_orig.png"
    )

    # Save inpainted disparity map as npy and image file
    disparity_inpainted = cv2.inpaint(
        disp_8bit,
        invalid_disp_mask.astype(np.uint8) * 255,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )
    disp_inpainted_orig_scale = cv2.normalize(
        disparity_inpainted,
        None,
        alpha=(disparity[disparity > 0]).min(),
        beta=disparity.max(),
        norm_type=cv2.NORM_MINMAX,
    )
    np.save(disp_save_dir / f"{img_name}.npy", disp_inpainted_orig_scale)
    disparity_inpainted_colored = cv2.applyColorMap(
        np.uint8(disparity_inpainted), cv2.COLORMAP_JET
    )
    Image.fromarray(cv2.cvtColor(disparity_inpainted_colored, cv2.COLOR_BGR2RGB)).save(
        disp_save_dir / f"{img_name}.png"
    )

    # Compute the depth map and save
    np.save(
        depth_save_dir / f"{img_name}.npy",
        (focal_length * baseline) / disp_inpainted_orig_scale,
    )


def main(left_img_dir, right_img_dir, output_dir, cam_info_file):
    left_imgs, right_imgs = load_image_list(left_img_dir, right_img_dir)

    disp_save_dir = Path(output_dir) / f"disp_maps"
    disp_save_dir.mkdir(parents=True, exist_ok=True)

    depth_save_dir = Path(output_dir) / f"depth_maps"
    depth_save_dir.mkdir(parents=True, exist_ok=True)

    cam_matrix = np.load(cam_info_file)
    focal_length = cam_matrix[0, 0]
    baseline = -cam_matrix[0, 3] / focal_length

    stereo_matcher = create_stereo_matcher()

    for img_name, left_path in tqdm(left_imgs.items(), total=len(left_imgs)):
        if img_name not in right_imgs:
            raise ValueError(f"Matching right image not found for {img_name}")
        process_image(
            img_name,
            left_path,
            right_imgs[img_name],
            disp_save_dir,
            depth_save_dir,
            focal_length,
            baseline,
            stereo_matcher,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--parent_output_dir",
        type=str,
        required=True,
        help="Parent output directory",
    )
    parser.add_argument(
        "--bag_file_name",
        type=str,
        required=True,
        help="Name of the bag file",
    )

    args = parser.parse_args()

    output_dir = Path(args.parent_output_dir) / args.bag_file_name
    left_img_path = output_dir / "front_cam/left"
    right_img_path = output_dir / "front_cam/right"
    cam_info_file = output_dir / "right_cam_info.npy"

    main(left_img_path, right_img_path, output_dir, cam_info_file)
