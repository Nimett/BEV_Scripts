import io
import glob
from pathlib import Path
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
from transformations import euler_from_quaternion, euler_matrix


def euclidean_to_homogeneous(euclidean_coords):
    """Convert Euclidean coordinates to homogeneous coordinates by adding a row of ones."""
    euclidean_coords = np.asarray(euclidean_coords)
    ones_row = np.ones((1, *euclidean_coords.shape[1:]), dtype=euclidean_coords.dtype)
    homogeneous_coords = np.concatenate((euclidean_coords, ones_row), axis=0)
    return homogeneous_coords


def pose_to_transformation_matrix(pose):
    """Convert a pose (position + quaternion) to a 3x4 transformation matrix."""
    pose = np.asarray(pose)
    position = pose[:3]
    quaternion = pose[3:]

    euler_angles = euler_from_quaternion(
        [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
    )
    rotation_matrix = euler_matrix(*euler_angles)[:3, :3]

    transformation_matrix = np.hstack((rotation_matrix, position.reshape(3, 1)))
    return transformation_matrix


def backproject_points(cam_intrinsics, pixel_coords, depths, ext_matrix, image_height):
    """Back-project 2D image points into 3D space using depth, intrinsics, and extrinsics."""
    cam_intrinsics = np.asarray(cam_intrinsics)
    pixel_coords = np.asarray(pixel_coords)
    depths = np.asarray(depths)

    # Invert intrinsics and unproject to normalized device coordinates
    cam_intrinsics_inv = np.linalg.inv(cam_intrinsics)
    homogeneous_pixels = euclidean_to_homogeneous(pixel_coords)
    rays = cam_intrinsics_inv @ homogeneous_pixels

    # Normalize rays so z-component is 1
    # rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    rays /= rays[2, :]

    # Scale rays by depth to get 3D points in camera frame
    depths_flat = depths.flatten()
    points_camera = rays * depths_flat

    # Apply extrinsic transformation
    points_camera_hom = euclidean_to_homogeneous(points_camera)
    points_external = ext_matrix[:-1, :] @ points_camera_hom

    # Reshape points back to (3, W, H)
    return points_camera.reshape(3, -1, image_height), points_external.reshape(
        3, -1, image_height
    )


def morphological_cleanup(
    image, kernel_size=3, dilate_iters=5, erode_iters=1, threshold_value=127, apply=True
):
    """Apply erosion and dilation to a grayscale image."""
    if image.ndim != 2:
        raise ValueError("Input image must be a single-channel grayscale image.")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Apply erosion and dilation
    if apply:
        eroded = cv2.erode(image, kernel, iterations=erode_iters)
        dilated = cv2.dilate(eroded, kernel, iterations=dilate_iters)
        final_eroded = cv2.erode(dilated, kernel, iterations=erode_iters)
    else:
        final_eroded = image

    # Apply thresholding
    _, binary_image = cv2.threshold(
        final_eroded, threshold_value, 255, cv2.THRESH_BINARY
    )

    return binary_image


def setup_plot(fig_size, dpi):
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.xticks([])
    plt.yticks([])
    return fig, ax


def compute_relative_ext_matrices(ext_mat):
    rel_ext = [np.vstack((np.hstack((np.eye(3), np.zeros((3, 1)))), [0, 0, 0, 1]))]
    base_inv = np.linalg.inv(ext_mat[0])
    rel_ext += [base_inv @ ext for ext in ext_mat[1:]]
    return rel_ext


def prepare_for_dilation(
    depth_maps, ext_mat, masks, labels, seg_classes, cam_par, fig_size, dpi
):
    img_h, img_w = masks[0].shape[1], masks[0].shape[2]
    x, y = np.meshgrid(np.arange(img_w), np.arange(img_h))
    pixel_coords = np.array([x.flatten(), y.flatten()])

    rel_ext = compute_relative_ext_matrices(ext_mat)
    list_of_buffers = []

    for seg_class_name, seg_class_info in seg_classes.items():
        fig, ax = setup_plot(fig_size, dpi)

        for i, (depth_map, ext, mask_set, label_set) in enumerate(
            zip(depth_maps, rel_ext, masks, labels)
        ):
            points, points_ext = backproject_points(
                cam_par, pixel_coords, depth_map, ext, img_h
            )
            points_ext = points_ext.reshape(3, -1)

            for m, l in zip(mask_set, label_set):
                curr_label = l[:-5].strip()
                if curr_label == seg_class_name:
                    if not seg_class_info["active"] or i == 0:
                        mask_flat = m.flatten()
                        ax.scatter(
                            points_ext[0][mask_flat],
                            points_ext[2][mask_flat],
                            s=1,
                            color=[1, 1, 1],
                            label=curr_label,
                        )

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        list_of_buffers.append(io.BytesIO(buf.getvalue()))
        plt.close(fig)

    return list_of_buffers


def load_segmentation_classes(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def load_depth_maps(folder):
    files = sorted(glob.glob(f"{folder}/*.npy"))
    return [np.load(f) for f in files]


def load_masks_or_labels(folder, extension):
    files = sorted(glob.glob(f"{folder}/*.{extension}.npy"))
    return [np.load(f) for f in files]


def load_extrinsics(filepath):
    with open(filepath, "rb") as file:
        extrinsics = pickle.load(file)
    ext_mat = [
        np.vstack((pose_to_transformation_matrix(pose), [0, 0, 0, 1]))
        for pose in extrinsics.values()
    ]
    return list(extrinsics.keys()), ext_mat


def setup_output_dirs(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def process_image(seg_classes, list_of_buffers, tmp_img_folder, fig_size, dpi):
    coloured_img = np.ones((fig_size * dpi, fig_size * dpi, 3), np.uint8) * 255
    coded_img = np.zeros((fig_size * dpi, fig_size * dpi, len(seg_classes) + 1))

    for i, (seg_class, img_buffer) in enumerate(zip(seg_classes, list_of_buffers)):
        tmp_img_path = tmp_img_folder / f"tmp_{i}.png"
        Image.open(img_buffer).save(str(tmp_img_path))

        img_for_morp = cv2.imread(str(tmp_img_path), cv2.IMREAD_GRAYSCALE)
        binary_img = morphological_cleanup(
            img_for_morp, apply=not seg_classes[seg_class]["active"]
        )

        if not seg_classes[seg_class]["active"]:
            colour = [float(c) for c in seg_classes[seg_class]["colour"].split(",")]
            coloured_img[binary_img == 255] = [x * 255 for x in colour[::-1]]
            coded_img[binary_img == 255, seg_classes[seg_class]["id"]] = seg_classes[
                seg_class
            ]["id"]
        else:
            process_circles(
                binary_img, coloured_img, coded_img, seg_classes[seg_class]["id"]
            )

    mask = np.all(coded_img == 0, axis=-1)
    coded_img[mask, 0] = 255

    return coloured_img, coded_img


def process_circles(binary_img, coloured_img, coded_img, class_id):
    circle_radius = 13
    circle_color = (0, 0, 255)
    circle_thickness = -1

    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

            cv2.circle(
                coloured_img,
                (centroid_x, centroid_y),
                circle_radius,
                circle_color,
                circle_thickness,
            )

            mask = np.zeros(coded_img.shape[:2], dtype=np.uint8)
            cv2.circle(
                mask, (centroid_x, centroid_y), circle_radius, 1, circle_thickness
            )
            coded_img[mask == 1, class_id] = class_id


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--seg_class_file",
        help="YAML file with segmentation classes",
        default="seg_classes.yaml",
    )
    parser.add_argument(
        "--parent_output_dir",
        help="Parent output directory",
    )
    parser.add_argument(
        "--bag_file_name",
        help="Bag file name",
    )
    args = parser.parse_args()

    fig_size, dpi = 6, 100
    n_accumulated = 10
    camera = "front_cam"

    working_folder = Path(args.parent_output_dir) / args.bag_file_name
    tmp_img_folder = Path.cwd() / ".tmp"
    setup_output_dirs(tmp_img_folder)
    setup_output_dirs(working_folder / "bev_maps")

    seg_classes = load_segmentation_classes(args.seg_class_file)
    cam_par = np.load(Path(args.parent_output_dir) / f"{camera}_info.npy")

    depth_maps = load_depth_maps(working_folder / "depth_maps")
    masks = load_masks_or_labels(working_folder / "segmentation", "mask")
    labels = load_masks_or_labels(working_folder / "segmentation", "label")
    timestamps, ext_mat = load_extrinsics(working_folder / "time_transform_file.pkl")

    pbar = tqdm(total=len(depth_maps))
    start = 0

    while start + n_accumulated < len(depth_maps):
        img_name = timestamps[start]

        curr_depth_maps = depth_maps[start : start + n_accumulated]
        curr_ext_mat = ext_mat[start : start + n_accumulated]
        curr_masks = masks[start : start + n_accumulated]
        curr_labels = labels[start : start + n_accumulated]

        list_of_buffers = prepare_for_dilation(
            curr_depth_maps,
            curr_ext_mat,
            curr_masks,
            curr_labels,
            seg_classes,
            cam_par[:, :-1],
            fig_size,
            dpi,
        )

        coloured_img, coded_img = process_image(
            seg_classes, list_of_buffers, tmp_img_folder, fig_size, dpi
        )

        cv2.imwrite(str(working_folder / "bev_maps" / f"{img_name}.png"), coloured_img)
        np.save(str(working_folder / "bev_maps" / f"{img_name}.npy"), coded_img)

        start += 1
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
