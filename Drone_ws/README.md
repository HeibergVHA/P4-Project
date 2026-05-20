## How to set up docker with px4_simulation

Assuming P4-Project was cloned into ~/Documents, run the docker using the command:

```
docker run -it --rm --name drone_container \
-v ~/Documents/P4-Project/Drone_ws/src:/ros2_ws/src \
--device=/dev/ttyAMA0 \
--network=host \
drone bash
```

Install gnome-terminal (multiple processes must be done at once).

```
sudo apt install gnome-terminal
```

Then, to install separate dependencies, do the following sequence:

```
cd deps

git clone https://github.com/PX4/px4_msgs.git

git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git

cd Micro-XRCE-DDS-Agent

mkdir build && cd build

cmake ..

make -j$(nproc)

sudo make install

sudo ldconfig /usr/local/lib/

```

Now go back to ros2_ws and build the ros2 packages

``` 
cd /ros2_ws

colcon build
```

