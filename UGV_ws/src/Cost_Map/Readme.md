
# To run the node 

cd~UGV_ws

colcon build

source install/setup.bash

ros2 run Cost_Map lidar_costmap_node

# bug fix 
Only If there is problems with finding the services or "the passed service type is invalid"
Use: (if it doesn't fix it then it's a different problem)

source ~/(whatever-the-folder-is-in)/P4-Project/UGV_ws/install/setup.bash

# To visualise the cost map. (it's own terminal after running the node)

Run:      rviz2 


# The two Services: (it's own terminal)
# For the Cost Map service use to create a new cost map: 

ros2 service call /generate_costmap std_srvs/srv/Trigger "{}"

# To show threshold interfaces use :

ros2 interface show threshold_interfaces/srv/SetThresholds

# To Change the thresholds use the below command and change the values to your liking:
 Note:
 If u don't put in values for them they will defualt to 0.
 Also no restart needed you can change them live and they do have a defult in the code they use normaly. 

ros2 service call /set_thresholds threshold_interfaces/srv/SetThresholds \
"{flat_caution_threshold: 0.03, caution_obstacle_threshold: 0.08, map_resolution: 0.05, 
inflation_radius_meters: 0.05, max_filter_size: 2, min_filter_size: 2}"


# To load the numpy.py file output for the coordinates use:

data = np.load("path/to/costmap_<timestamp>.npy")

# data[:, 0] → x  |  data[:, 1] → y  |  data[:, 2] → cost

# The function" load_pcd_and_process "on line 115 
 This has the transformation function from line: 126-132
 Make sure you visualise the costmap to confirm if you have the the right transformation or just disable it. 


# Below is the Arciteecture of the code 
```
┌─────────────┐
│   PCD File  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│           load_pcd_and_process()            │
│      Rotate 90° around Y axis               │
│      (sensor frame → map frame)             │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│          compute_elevation_map()            │
│      Average Z value per grid cell          │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│   terrain_roughness = max_filter(elevation) │
│                     - min_filter(elevation) │
└──────┬──────────────────────────────────────┘
       │
       ├──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼
┌──────────────┐   ┌─────────────────────┐   ┌──────────────────┐
│ Static Layer │   │  Inflation Layer    │   │  Master Layer    │
│              │   │                     │   │                  │
│ free    = 10 │   │ OpenCV dilation of  │   │ element-wise     │
│ caution = 50 │   │ lethal cells        │   │ max(static,      │
│ lethal  = 100│   │ buffer cost = 90    │   │     inflation)   │
│ unknown = -1 │   │                     │   │                  │
└──────┬───────┘   └──────────┬──────────┘   └────────┬─────────┘
       │                      │                       │
       ▼                      ▼                       ▼
/static_costmap     /inflation_costmap        /master_costmap
                                                      │
                                                      ▼
                                             costmap_<timestamp>.npy
```