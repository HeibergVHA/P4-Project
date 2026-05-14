import open3d as o3d
import numpy as np

# ملفات الإدخال والإخراج
INPUT_FILE = "WALLR3.pcd"
OUTPUT_FILE = "rc_car_template.pcd"

# حدود الصندوق (عدّلها حسب بياناتك)
# [xmin, ymin, zmin]
min_bound = np.array([4.6, -0.5, -0.5])

# [xmax, ymax, zmax]
max_bound = np.array([5.2, 1.0, 1.0])

def remove_ground(pcd):
    print("Removing ground plane...")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.15,
        ransac_n=3,
        num_iterations=1000
    )

    ground = pcd.select_by_index(inliers)
    objects = pcd.select_by_index(inliers, invert=True)

    return objects


def extract_largest_cluster(pcd):
    print("Clustering objects...")

    labels = np.array(
        pcd.cluster_dbscan(
            eps=0.05,      # distance threshold (tune!)
            min_points=30
        )
    )

    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")

    largest_cluster = None
    max_points = 0

    for i in range(max_label + 1):
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        if len(cluster.points) > max_points:
            largest_cluster = cluster
            max_points = len(cluster.points)

    return largest_cluster


def main():
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)

    print("Cropping point cloud...")

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped = pcd.crop(bbox)

    print(f"Original points: {len(pcd.points)}")
    print(f"Cropped points: {len(cropped.points)}")

    # Step 1: remove ground
    objects = remove_ground(cropped)

    print(f"Points after ground removal: {len(objects.points)}")

    # Step 2: cluster
    rc_car = extract_largest_cluster(objects)

    print(f"Points in largest cluster: {len(rc_car.points)}")

    print("Saving filtered point cloud...")
    o3d.io.write_point_cloud(OUTPUT_FILE, rc_car)

    print("Done.")


if __name__ == "__main__":
    main()