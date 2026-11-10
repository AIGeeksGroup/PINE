# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import trimesh


def save_generated_gt_point_cloud(gen_cloud, gen_color, gt_cloud, gt_color, filename="comparison.ply", spacing=None, scale_factor=0.2, axis=0, mask=None, gt_mask=None):
    """
    Combines exactly two HxWx3 point clouds (Generated + GT) into a single PLY file.
    Applies offset based on center-to-center distance.

    Args:
        gen_cloud (np.array): Generated cloud, shape (H, W, 3).
        gen_color (np.array): Generated colors, shape (H, W, 3).
        gt_cloud (np.array): Ground Truth cloud, shape (H, W, 3).
        gt_color (np.array): Ground Truth colors, shape (H, W, 3).
        filename (str): Output file path.
        spacing (float or None):
            If Float: Target DISTANCE between centers.
            If None: Auto-calculated to ensure no overlap with padding.
        scale_factor (float or None):
            Relative padding size (gap = average_width * scale_factor).
            If None, auto-selects based on object size.
        axis (int): Axis to apply offset (0=X, 1=Y, 2=Z).
        mask (np.array or None): Optional (H, W) bool array for gen cloud — True = keep.
        gt_mask (np.array or None): Optional (H, W) bool array for gt cloud — True = keep.
    """

    # 1. Flatten Data
    gen_pts = gen_cloud.reshape(-1, 3)
    gen_rgb = gen_color.reshape(-1, 3)
    gt_pts = gt_cloud.reshape(-1, 3)
    gt_rgb = gt_color.reshape(-1, 3)

    if mask is not None:
        flat_mask = mask.reshape(-1)
        gen_pts = gen_pts[flat_mask]
        gen_rgb = gen_rgb[flat_mask]
    if gt_mask is not None:
        flat_gt_mask = gt_mask.reshape(-1)
        gt_pts = gt_pts[flat_gt_mask]
        gt_rgb = gt_rgb[flat_gt_mask]

    if len(gen_pts) == 0 or len(gt_pts) == 0:
        print("Error: One of the point clouds is empty.")
        return

    # 2. Calculate Dimensions & Centers
    # We need min/max and center on the specific axis we are stacking along
    gen_min, gen_max = gen_pts[:, axis].min(), gen_pts[:, axis].max()
    gt_min, gt_max = gt_pts[:, axis].min(), gt_pts[:, axis].max()
    
    gen_width = gen_max - gen_min
    gt_width = gt_max - gt_min
    avg_width = (gen_width + gt_width) / 2.0
    
    gen_center = (gen_min + gen_max) / 2.0
    gt_center = (gt_min + gt_max) / 2.0

    # 3. Determine Target Center Distance
    dist_between_centers = 0.0

    if spacing is not None:
        # User specified fixed distance between centers
        dist_between_centers = spacing
    else:
        # Auto-calculate based on size
        if scale_factor is None:
            # If objects are small (e.g. normalized coords < 5.0), use bigger relative gap
            scale_factor = 0.5 if avg_width < 5.0 else 0.2
            
        # We want: Center_Dist = Half_W_Gen + Half_W_GT + Gap
        gap = avg_width * scale_factor
        dist_between_centers = (gen_width / 2.0) + (gt_width / 2.0) + gap

    # 4. Apply Offsets
    # Target GT Center = Gen Center + Distance
    target_gt_center = gen_center + dist_between_centers
    offset_val = target_gt_center - gt_center
    
    translation = np.zeros(3)
    translation[axis] = offset_val
    
    translated_gt_pts = gt_pts + translation
    
    # 5. Combine
    combined_pts = np.vstack([gen_pts, translated_gt_pts])
    
    # Handle Colors (Ensure uint8)
    def to_uint8(c):
        if c.dtype.kind == 'f':
            return (c * 255).astype(np.uint8)
        return c.astype(np.uint8)

    combined_colors = np.vstack([to_uint8(gen_rgb), to_uint8(gt_rgb)])

    # 6. Save
    pcd = trimesh.PointCloud(vertices=combined_pts, colors=combined_colors)

    # 2. Define a "flip" matrix (OpenCV -> OpenGL/Standard 3D)
    # This flips Y and Z to make Y-up and Z-out
    flip_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # 3. Apply the transform
    pcd.apply_transform(flip_transform)

    pcd.export(filename)


def depth_flying_points(depth, rtol=0.04, kernel_size=3, mid_threshold=0.5):
    """
    Flags "flying point" pixels — those whose depth is intermediate between the
    local foreground and background depth.

    Pixels at true depth discontinuities (fine-grained edges) whose depth is
    close to either the local min or max are NOT flagged, since they represent
    real foreground/background boundaries predicted by the model.

    Args:
        depth: (H, W) numpy array
        rtol: Relative tolerance for depth edge detection (default 0.04)
        kernel_size: Neighborhood size (default 3)
        mid_threshold: Normalized distance from midpoint below which a pixel
                       is considered intermediate (flying point). Default 0.5
                       means: depth within [min + 0.25*range, max - 0.25*range]
                       is a flying point. Depth close to min or max is a real edge.

    Returns:
        (H, W) bool array — True only for flying points (not fine-grained edges)
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    depth_clean = np.where(np.isfinite(depth), depth, 0.0)
    local_max = maximum_filter(depth_clean, size=kernel_size)
    local_min = minimum_filter(depth_clean, size=kernel_size)

    # Step 1: detect all depth discontinuities (same as depth_edge)
    rel_range = np.zeros_like(depth)
    valid = local_max > 0
    np.divide(local_max - local_min, local_max, out=rel_range, where=valid)
    is_edge = rel_range > rtol

    # Step 2: among edge pixels, check if depth is intermediate (flying point)
    # or close to one extreme (fine-grained edge).
    # Normalize distance from midpoint: 0 = at midpoint, 1 = at an extreme.
    local_range = local_max - local_min
    mid_depth = (local_max + local_min) / 2.0
    dist_from_mid = np.abs(depth_clean - mid_depth)
    normalized_dist = np.ones_like(dist_from_mid)
    valid_range = local_range > 0
    np.divide(dist_from_mid, local_range / 2.0, out=normalized_dist, where=valid_range)

    # Flying point: depth closer to midpoint than to either extreme
    is_flying_point = normalized_dist < mid_threshold

    return is_edge & is_flying_point


def save_single_point_cloud(points, colors, filename, transform_to_gl=True, mask=None):
    """
    Save a single point cloud as PLY file.

    Args:
        points: (H, W, 3) or (3, H, W) NumPy array of 3D points
        colors: (H, W, 3) NumPy array of RGB colors in [0, 1] range
        filename: Output .ply file path
        transform_to_gl: If True, apply OpenGL coordinate transform (flip Y and Z)
        mask: Optional (H, W) bool array — True = keep pixel, False = discard
    """
    # Handle both (H, W, 3) and (3, H, W) input formats
    if points.shape[0] == 3 and len(points.shape) == 3:
        # Input is (3, H, W), transpose to (H, W, 3)
        points = points.transpose(1, 2, 0)

    # Flatten to (N, 3)
    flat_points = points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)

    # Apply spatial mask (e.g. to remove depth-edge flying points)
    if mask is not None:
        flat_mask = mask.reshape(-1)
        flat_points = flat_points[flat_mask]
        flat_colors = flat_colors[flat_mask]

    if len(flat_points) == 0:
        print(f"Warning: No points to save to {filename}")
        return

    # Ensure colors are uint8 0-255 for trimesh
    if flat_colors.dtype.kind == 'f':
        if flat_colors.max() <= 1.0 + 1e-5:
            flat_colors = (flat_colors * 255).astype(np.uint8)
        else:
            flat_colors = flat_colors.astype(np.uint8)

    # Apply coordinate system transformation if requested
    if transform_to_gl:
        flat_points = flat_points.copy()
        flat_points[:, 1] = -flat_points[:, 1]  # Flip Y
        flat_points[:, 2] = -flat_points[:, 2]  # Flip Z

    # Create and export point cloud
    pcd = trimesh.PointCloud(flat_points, colors=flat_colors)
    pcd.export(filename)

