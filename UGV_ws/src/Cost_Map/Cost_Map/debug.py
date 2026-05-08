# debug_match.py  — run standalone, no ROS needed
import open3d as o3d
import numpy as np

TEMPLATE = "src/Cost_Map/resource/rc_car_template.pcd"
SCENE = "src/Cost_Map/resource/WALLR1.pcd"  # ← fix this to real path
VOXEL    = 0.05

def preprocess(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    # Always recompute normals — don't rely on them being in the file
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    down.orient_normals_consistent_tangent_plane(k=15)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5, max_nn=100))
    return down, fpfh

tmpl = o3d.io.read_point_cloud(TEMPLATE)
scene = o3d.io.read_point_cloud(SCENE)

# Check bounding boxes — are they even in the same coordinate range?
print("Template bbox:", tmpl.get_axis_aligned_bounding_box())
print("Scene bbox:   ", scene.get_axis_aligned_bounding_box())

# Visualise both together before any alignment
tmpl.paint_uniform_color([1, 0, 0])   # red = template
scene.paint_uniform_color([0, 1, 0])  # green = scene
o3d.visualization.draw_geometries([tmpl, scene], window_name="Before alignment")

t_down, t_fpfh = preprocess(tmpl, VOXEL)
s_down, s_fpfh = preprocess(scene, VOXEL)

ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    t_down, s_down, t_fpfh, s_fpfh, mutual_filter=True,
    max_correspondence_distance=VOXEL*1.5,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL*1.5),
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
)
print("RANSAC fitness:", ransac.fitness, " T:\n", ransac.transformation)

icp = o3d.pipelines.registration.registration_icp(
    t_down, s_down, VOXEL*0.4, ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print("ICP fitness:", icp.fitness, "  rmse:", icp.inlier_rmse)

tmpl_aligned = o3d.geometry.PointCloud(t_down)
tmpl_aligned.transform(icp.transformation)
tmpl_aligned.paint_uniform_color([1, 0, 0])
s_down.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([tmpl_aligned, s_down], window_name="After alignment")