
""""""""""""""""""""""""""""""""
To run node: 

cd --- UGV_ws

colcon build

source install/setup.bash

ros2 run Cost_Map lidar_costmap_node

(bug)
If there is problems with finding the services or "the passed service type is invalid"
Use: (if it doesn't fix it then there is a different problem)

source ~/(whatever-the-folder-is-in)/P4-Project/UGV_ws/install/setup.bash
""""""""""""""""""""""""""""""""""

To visualise the cost map. (it's own terminal)

Run rviz2 ( do this after running the node)

Command: rviz2


"""""""""""""""""""""""""""""""""""
Services: (it's own terminal)
For the Cost Map service use to create a new cost map: 

ros2 service call /generate_costmap std_srvs/srv/Trigger "{}"




To show threshold interfaces use :

ros2 interface show threshold_interfaces/srv/SetThresholds

To Change the thresholds use this format: (Note if u don't but in values for them they will defualt to 0)

ros2 service call /set_thresholds threshold_interfaces/srv/SetThresholds \
"{flat_caution_threshold: 0.03, caution_obstacle_threshold: 0.08, map_resolution: 0.05, 
inflation_radius_meters: 0.05, max_filter_size: 2, min_filter_size: 2}"


""""""""""""""""""""""""""""""""""""""

To load the numpy.py file for the coordinates use:

data = np.load("path/to/costmap_20250421_143022.npy")


 """""""""""""""""""""""""""""""""""""""""""""""""""
 For dockerfile 

#Build it (Be in workspace)

 docker build -t costmap_test .

 docker run -v ~/Desktop/P4-Project/UGV_ws:/ros2_ws costmap_test bash

