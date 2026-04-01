#!/bin/bash
source /opt/ros/jazzy/setup.bash

# Also source your workspace if it's been built
if [ -f /ros2_ws/install/setup.bash ]; then
  source /ros2_ws/install/setup.bash
fi

exec "$@"