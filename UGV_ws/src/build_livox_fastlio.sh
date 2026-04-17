#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
cd /ros2_ws

echo "Building livox_sdk_vendor..."
colcon build --packages-select livox_sdk_vendor \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash

echo "Building livox_interfaces..."
colcon build --packages-select livox_interfaces \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash

echo "Building livox_ros2_driver..."
colcon build --packages-select livox_ros2_driver \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash

echo "Building fast_lio..."
colcon build --packages-select fast_lio \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "All builds complete."