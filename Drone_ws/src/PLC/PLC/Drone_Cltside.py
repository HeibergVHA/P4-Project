#!/usr/bin/env python3

import socket
import struct
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger
import zipfile
import os
import io


class Drone_Cltside(Node):
    """
    ROS 2 TCP client node.

    1. Connects to the TcpMeanServerNode on startup.
    2. Receives a numpy array frame from the server.
    3. Computes the mean of the array.
    4. Sends the mean back as a 4-byte float32 LE.
    5. Publishes the received array and computed mean to ROS topics.

    Wire protocol - server to client (numpy frame):
        Unpsecified as of yet, but will contain commands
        form of multiple {x,y,z} coordinates in a scan pattern

    Wire protocol - client to server (mean reply):
        [8 Bytes] - file size as uint64 LE

    Parameters:
        host (str) - server IP address  [default: '127.0.0.1']
        port (int) - server port        [default: 12347]

    Topics published:
        ~/array (std_msgs/Float32MultiArray) - array received from server
        ~/mean  (std_msgs/Float32)           - computed mean sent back

    Services:
        connect    (std_srvs/Trigger) - (re)connect to the server
        disconnect (std_srvs/Trigger) - disconnect from the server
        send   (std_srvs/Trigger) - zip and send the bag to the server
    """

    def __init__(self):
        """
        
        
        
        
        """
        super().__init__('drone_Client_side')

        # parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 12347)
        self.declare_parameter('bag_path', '/ros2_ws/bags/scan_20240617_153000') # default bag path, can be overridden by parameter

        # state
        self._sock: socket.socket | None = None
        self._connected = False
        self._lock = threading.Lock()
        self.bag_name = None

        # services
        self.create_service(
            Trigger, 
            '/connect',
            self._cb_connect
        )
        
        self.create_service(
            Trigger, 
            '/disconnect', 
            self._cb_disconnect
        )
        
        self.create_service(
            Trigger,
            '/send',
            self._cb_send
        )

        self.start_recording_client = self.create_client(Trigger, '/start_recording')
        # connect on startup
        self.stop_recording_client = self.create_client(Trigger,'/stop_recording')
        self._connect()

    # connection helpers

    def _connect(self):
        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value

        if self._connected:
            return False, 'Already connected.'

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # socket.AF_INET, socket.SOCK_STREAM
            self.get_logger().info(f'Attempting to connect {host}:{port}...')
            sock.connect((host, port))
            with self._lock:
                self._sock      = sock
                self._connected = True
            self.get_logger().info(f'Connected to {host}:{port}')
            self.trigger_start_recording()
            return True, f'Connected to {host}:{port}'
        except Exception as e:
            self.get_logger().error(f'Connection failed: {e}')
            return False, str(e)

    def _disconnect(self):
        with self._lock:
            if not self._connected:
                return False, 'Not connected.'
            self._connected = False
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self.get_logger().info('Disconnected.')
        return True, 'Disconnected.'

    
    # Bag sending

    def _zip_bag(self, bag_path: str, zip_path: str) -> bytes:
        """
        Zips the entire rosbag directory into a file on disk rather than into ram

        rosbag2 saves bags as a directory containing:
          - scan_TIMESTAMP.db3      (the actual data)
          - metadata.yaml           (topic info, timestamps etc.)

        We zip the whole directory so both files travel together and the
        server can unzip and play it back directly with ros2 bag play.

        Returns the raw bytes of the zip file.
        """
    
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # os.walk() traverses the directory tree
            # root  = current directory path
            # dirs  = subdirectory names in root
            # files = file names in root
            for root, dirs, files in os.walk(bag_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    # arcname is the path stored inside the zip
                    # we make it relative so it unzips cleanly
                    arcname = os.path.relpath(full_path, os.path.dirname(bag_path))
                    zf.write(full_path, arcname)
                    self.get_logger().info(f'Zipping: {arcname}')

    def _send_bag(self):
        """
        Protocol:
          1. zip the bag directory into memory
          2. send 8 bytes: the zip file size as uint64 little-endian
             - this tells the server exactly how many bytes to read
          3. send the zip bytes in 4096-byte chunks
             - chunking prevents memory issues with large bags
        """
        if not self._connected:
            return False, 'Not connected.'

        bag_path = self.bag_name
        # Check the bag directory actually exists before trying to send
        if not os.path.isdir(bag_path):
            return False, f'Bag path does not exist: {bag_path}'
        
        #Write zip next to the bag directory.
        zip_path = bag_path.rstrip('/') + '_transfer.zip'

        try:
            self.get_logger().info(f'Zipping bag: {bag_path} to {zip_path}...')
            self._zip_bag(bag_path, zip_path)

            zip_size = os.path.getsize(zip_path)
            self.get_logger().info(f'Sending {zip_size} bytes to server...')

            with self._lock:
                if not self._connected or self._sock is None:
                    return False, 'Lost connection.'

                # First mesage is the size of the zip file, where the least significant byte is first (little-endian)
                # '<Q' means: < = little-endian, Q = unsigned 64-bit integer
                self._sock.sendall(struct.pack('<Q', zip_size))

            # Step 2: send the zip data in 4096-byte chunks
            # sending in chunks is safer than one giant sendall() apparently
            # for large bags (hundreds of MB)
            chunk_size = 4096
            sent = 0

            with open(zip_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    with self._lock:
                        self._sock.sendall(chunk)
                    sent += len(chunk)

                    if sent % (10 * 1024 * 1024) <chunk_size:  # Log every 10 MB sent
                        pct = (sent / zip_size) * 100
                        self.get_logger().info(f'Sent {pct:.1f}%')

            self.get_logger().info(f'Bag sent successfully: {zip_size} bytes')
            return True, f'Sent {sent} bytes'

        except Exception as e:
            self.get_logger().error(f'Send failed: {e}')
            self._connected = False
            return False, str(e)
        finally:
            # Clean up the zip file after sending
            if os.path.exists(zip_path):
                os.remove(zip_path)
                self.get_logger().info(f'Removed temporary zip file: ')
        
    def _send_in_thread(self, response):
        success, message = self._send_bag()
        self.get_logger().info(f'Send result: {message}')
    # service callbacks

    def _cb_send(self, _request, response):
        threading.Thread(target=self._send_in_thread, args=(response,), daemon=True).start()
        response.success = True
        response.message = 'Send started in background'
        return response
    
    def _cb_connect(self, _request, response):
        response.success, response.message = self._connect()
        return response

    def _cb_disconnect(self, _request, response):
        response.success, response.message = self._disconnect()
        return response


    def destroy_node(self):
        self._disconnect()
        super().destroy_node()

    def trigger_start_recording(self):
        request = Trigger.Request()
        future = self.start_recording_client.call_async(request)
        future.add_done_callback(self.start_recording_response_cb)

    def start_recording_response_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info(f'Start recording triggered successfully: {result.message}')
            self.bag_name = result.message
            self.recording_timer = self.create_timer(20.0, self.trigger_stop_recording)
        else:
            self.get_logger().error(f'Failed to trigger start recording: {result.message}')

    def trigger_stop_recording(self):
        self.recording_timer.cancel()
        request = Trigger.Request()
        future = self.stop_recording_client.call_async(request)
        future.add_done_callback(self.stop_recording_response_cb)

    def stop_recording_response_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info(f'Stop recording triggered successfully: {result.message}')
        else:
            self.get_logger().error(f'Failed to trigger stop recording: {result.message}')

def main(args=None):
    rclpy.init(args=args)
    node = Drone_Cltside()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()