FROM ros:humble-ros-base
SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y \
    build-essential cmake git wget curl nano vim \
    python3-pip python3-colcon-common-extensions \
    python3-vcstool python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    libeigen3-dev libceres-dev libsuitesparse-dev \
    libyaml-cpp-dev libpcap-dev libgoogle-glog-dev \
    libgflags-dev libpcl-dev \
    ros-humble-pcl-conversions ros-humble-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    ros-humble-mavros ros-humble-mavros-extras ros-humble-mavros-msgs \
    && rm -rf /var/lib/apt/lists/*
RUN /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

RUN apt update && apt install -y \
    mesa-utils x11-apps ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    python3-pandas python3-future python3-serial python3-scipy \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c " \
    source /opt/ros/humble/setup.bash && \
    mkdir -p /ros2_ws/deps && \
    cd /ros2_ws/deps && \
    git clone https://github.com/HeibergVHA/livox_ros2_driver.git --depth 1 && \
    cd /ros2_ws && \
    colcon build --packages-select livox_sdk_vendor \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --event-handlers console_cohesion+ && \
    source /ros2_ws/install/setup.bash && \
    colcon build --packages-select livox_interfaces \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --event-handlers console_cohesion+ && \
    source /ros2_ws/install/setup.bash && \
    colcon build --packages-select livox_ros2_driver \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --event-handlers console_cohesion+"

ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV ROS_DOMAIN_ID=7
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

WORKDIR /ros2_ws
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
