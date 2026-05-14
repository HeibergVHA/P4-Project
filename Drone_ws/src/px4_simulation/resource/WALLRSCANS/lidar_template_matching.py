from tempfile import template

import open3d as o3d
import numpy as np
import copy


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

SCENE_FILE = "WALLR1.pcd"  # input scene containing the car
TEMPLATE_FILE = "rc_car_template.pcd"

VOXEL_SIZE = 0.05  # instead of 0.2  # meters (adjust for your data)


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsample and compute features.
    """

    print("Downsampling...")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    print("Estimating normals...")
    radius_normal = voxel_size * 2

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30
        )
    )

    print("Computing FPFH features...")
    radius_feature = voxel_size * 5

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )

    return pcd_down, fpfh


def execute_global_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    voxel_size
):
    """
    Coarse alignment using RANSAC feature matching.
    """

    distance_threshold = voxel_size * 1.0 # 1.5

    print("Running RANSAC global registration...")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,

        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),

        ransac_n=4,

        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            )
        ],

        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            4000000,
            500
        )
    )

    return result


def refine_registration(source, target, initial_transform, voxel_size):
    """
    Refine alignment using ICP.
    """

    distance_threshold = voxel_size * 0.4

    print("Running ICP refinement...")

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0, 0])  # red
    target_temp.paint_uniform_color([0, 1, 0])  # green

    source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp])

def check_physical_plausibility(transformation):
    """
    Reject impossible positions (e.g., underground, too far away).
    """

    translation = transformation[:3, 3]
    x, y, z = translation

    print("\n--- Position Check ---")
    print(f"Detected position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    # Example rules (adjust to your setup)
    if z < -0.2:
        print("❌ Rejected: Object below ground")
        return False

    if abs(x) > 50 or abs(y) > 50:
        print("❌ Rejected: Object too far away")
        return False

    print("✅ Position plausible")
    return True

def validate_match(result, template, scene, voxel_size):
    """
    Decide whether the template match is valid.
    """

    fitness = result.fitness
    rmse = result.inlier_rmse

    # Estimate number of inliers
    num_template_points = len(template.points)
    estimated_inliers = fitness * num_template_points

    print("\n--- Validation Metrics ---")
    print(f"Fitness: {fitness:.3f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Estimated inliers: {estimated_inliers:.0f}")

    # -------------------------------
    # THRESHOLDS (tune these!)
    # -------------------------------

    MIN_FITNESS = 0.3
    MAX_RMSE = voxel_size * 0.5
    MIN_INLIERS = 500

    # -------------------------------
    # VALIDATION LOGIC
    # -------------------------------

    if fitness < MIN_FITNESS:
        print("❌ Rejected: Low fitness")
        return False

    if rmse > MAX_RMSE:
        print("❌ Rejected: High RMSE")
        return False

    if estimated_inliers < MIN_INLIERS:
        print("❌ Rejected: Too few inliers")
        return False

    print("✅ Match accepted")
    return True

def validate_alignment_geometrically(template, scene, transformation, distance_threshold=0.1):
    """
    Check if transformed template actually aligns with scene points.
    """

    print("\n--- Geometric Validation ---")

    # Transform template
    template_transformed = copy.deepcopy(template)
    template_transformed.transform(transformation)

    # Build KD-tree for scene
    scene_kd_tree = o3d.geometry.KDTreeFlann(scene)

    distances = []

    # For each point in template, find nearest neighbor in scene
    for point in template_transformed.points:
        [_, idx, dists] = scene_kd_tree.search_knn_vector_3d(point, 1)

        if len(dists) > 0:
            distances.append(np.sqrt(dists[0]))

    distances = np.array(distances)

    if len(distances) == 0:
        print("❌ No correspondences found")
        return False

    # Metrics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    inlier_ratio = np.sum(distances < distance_threshold) / len(distances)

    print(f"Mean distance: {mean_dist:.4f}")
    print(f"Median distance: {median_dist:.4f}")
    print(f"Inlier ratio (<{distance_threshold} m): {inlier_ratio:.3f}")

    # -------------------------------
    # DECISION RULES (tune these!)
    # -------------------------------

    MIN_INLIER_RATIO = 0.5
    MAX_MEAN_DIST = distance_threshold

    if inlier_ratio < MIN_INLIER_RATIO:
        print("❌ Rejected: Poor overlap with scene")
        return False

    if mean_dist > MAX_MEAN_DIST:
        print("❌ Rejected: Points too far from scene")
        return False

    print("✅ Alignment is geometrically valid")
    return True

def visualize_local_match(template, scene, transformation, box_size=1.0):
    """
    Zoom into the detected location and show only nearby points.
    """

    print("\n--- Local Visualization ---")

    # Extract detected position
    center = transformation[:3, 3]
    print(f"Detected center: {center}")

    # Transform template into scene
    template_transformed = copy.deepcopy(template)
    template_transformed.transform(transformation)

    # Create bounding box around detected position
    half_size = box_size / 4.0
    min_bound = center - half_size
    max_bound = center + half_size

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Crop scene
    local_scene = scene.crop(bbox)

    print(f"Points in local region: {len(local_scene.points)}")

    # Color for visualization
    local_scene.paint_uniform_color([0.7, 0.7, 0.7])  # gray
    template_transformed.paint_uniform_color([1, 0, 0])  # red

    # Show only local region + template
    o3d.visualization.draw_geometries([
        local_scene,
        template_transformed,
        bbox
    ])

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    print("Loading point clouds...")

    scene = o3d.io.read_point_cloud(SCENE_FILE)
    template = o3d.io.read_point_cloud(TEMPLATE_FILE)

    print(f"Scene points: {len(scene.points)}")
    print(f"Template points: {len(template.points)}")

    # Preprocess
    scene_down, scene_fpfh = preprocess_point_cloud(scene, VOXEL_SIZE)
    template_down, template_fpfh = preprocess_point_cloud(template, VOXEL_SIZE)

    # Global registration
    result_ransac = execute_global_registration(
        template_down,
        scene_down,
        template_fpfh,
        scene_fpfh,
        VOXEL_SIZE
    )

    print("\nInitial transformation:")
    print(result_ransac.transformation)

    print(f"RANSAC fitness: {result_ransac.fitness}")
    print(f"RANSAC RMSE: {result_ransac.inlier_rmse}")

    # ICP refinement
    result_icp = refine_registration(
        template,
        scene,
        result_ransac.transformation,
        VOXEL_SIZE
    )

    # After ICP refinement
    result_icp = refine_registration(
        template,
        scene,
        result_ransac.transformation,
        VOXEL_SIZE
    )

    print("\nRefined transformation:")
    print(result_icp.transformation)

    # -----------------------------------
    # LOCAL INSPECTION (THIS IS YOUR GOAL)
    # -----------------------------------

    visualize_local_match(
        template,
        scene,
        result_icp.transformation,
        box_size=2.0  # adjust based on RC car size
    )

    """print("\nRefined transformation:")
    print(result_icp.transformation)

    print(f"ICP fitness: {result_icp.fitness}")
    print(f"ICP RMSE: {result_icp.inlier_rmse}")

    result_ransac_trans = np.array(result_ransac.transformation)
    print(result_ransac_trans[0:3, 3])
    min_bound = np.array(result_ransac_trans[0:3, 3]) - np.array([5.0, 5.0, 5.0]) # [xmin, ymin, zmin]
    max_bound = np.array(result_ransac_trans[0:3, 3]) + np.array([5.0, 5.0, 5.0]) # [xmax, ymax, zmax]

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped = scene.crop(bbox)

    # Visualize
    draw_registration_result(
        template,
        cropped,
        result_icp.transformation
    )

    o3d.io.write_point_cloud("result.pcd", cropped)"""


if __name__ == "__main__":
    main()