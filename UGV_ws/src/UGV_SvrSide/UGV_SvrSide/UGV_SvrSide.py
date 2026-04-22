#!/usr/bin/env python3

import socket
import struct
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger
import os
import zipfile


class UGV_SvrSide(Node):
    """
    ROS 2 TCP server node.

    1. Waits for one client to connect.
    2. Sends a random float32 array to the client on a timer.
    3. Reads the mean (4-byte float32) the client sends back.
    4. Publishes the received  mean to ~/mean.

    Wire protocol - server to client (numpy frame):
    [8 bytes] uint64 LE - number of bytes to follow
    [n bytes] the numpy array serialized as float32 LE

    Wire protocol - client to server 
    # Eventually we want to x,y,z coords

    Parameters:
        host          (str)   - bind address          [default: '0.0.0.0']
        port          (int)   - bind port              [default: 12347]
        save_path     (str)   - where to save received bags [default: '/ros2_ws/bags']

    Services:
        ~/disconnect (std_srvs/Trigger) - drop the current client
    """

    def __init__(self):
        super().__init__('ugv_server_side')

        # parameters
        self.declare_parameter('host',          '0.0.0.0')
        self.declare_parameter('port',          12347)
        self.declare_parameter('save_path',  '/ros2_ws/bags')



        # state
        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._connected  = False
        self._lock       = threading.Lock()

        # service
        self.create_service(Trigger, '/disconnect', self._cb_disconnect)

        self.start_reading_client = self.create_client(Trigger, '/start_reading')
        # start listening in background
        threading.Thread(target=self._listen_loop, daemon=True).start()

    # listen / accept loop

    def _listen_loop(self):
        """Listens for incoming client connections and handles them.
        This method runs in a separate thread and blocks
        
        host = Ip adress to bind to
        port = Port to bind to
        """

        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value

        # Create server socket (IPv4, SOCK_STREAM=TCP)
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Allow quick reuse of the address
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind and listen to the drone
        self._server_sock.bind((host, port))
        self._server_sock.listen(1)
        self.get_logger().info(f'Listening on {host}:{port} ...')

        while rclpy.ok(): # While ROS is running
            try:
                # Try to connect, Blocks until connection, and reads ip, port.
                conn, addr = self._server_sock.accept()
            except Exception as e: # If any error occurs
                if rclpy.ok():
                    self.get_logger().error(f'Accept error: {e}')
                break

            with self._lock: # Lock and Set client socket
                self._client_sock = conn
                self._connected   = True
            self.get_logger().info(f'Client connected: {addr}')

            self.receive_bag()

            self.get_logger().info('Waiting for next client ...')


    # -----------> Helpers <------------ #

    def _recv_exactly(self, n: int) -> bytes:
        """ 
        Receives exactly n bytes from the client socket
        """
        buf = b'' # b'' is an empty bytes object
        while len(buf) < n:
            chunk = self._client_sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError('Client closed the connection.')
            buf += chunk
        return buf
    
    def receive_bag(self):
        """
        Receives a zipped ros2 bag file from the client
        
        Protocol:
            1. Read 8 Bytes: the zip file size in bytes (uint64 LE)
            2. read exactly that many bytes in chunks, writing to disk
            3. Unzip the file to the save_path directory
            4. Delete the temporary zip file

        We write in chunks to avoid loadin the entire file in the RAM
        Which can cause memory issues for large bags
        """
        save_path = self.get_parameter('save_path').get_parameter_value().string_value

        # ---> Step 1 <--- #

        # Create a path
        os.makedirs(save_path, exist_ok=True)

        # Path to the temporary zip file
        zip_path = os.path.join(save_path, 'received_bag.zip')

        try:
            # Read 8 Bytes for file size
            size_bytes = self._recv_exactly(8)
            zip_size = struct.unpack('<Q', size_bytes)[0] # uint64 LE
            self.get_logger().info(f'Expecting {zip_size} bytes')

            # ---> Step 2 <--- #
            received = 0
            chunk_size = 4096

            with open(zip_path, 'wb') as f:
                while received < zip_size:
                    to_read = min(chunk_size, zip_size - received)
                    chunk = self._recv_exactly(to_read)

                    if not chunk:
                        raise ConnectionError('Client disconnected during transfer.')
                    
                    #Write the chunk to disk
                    f.write(chunk)
                    received += len(chunk)

                    # Log progress every 10MB
                    if received % (10 * 1024 * 1024) < chunk_size:
                        pct = (received / zip_size) * 100
                        self.get_logger().info(f'Received {pct:.1f}%')
            
            self.get_logger().info(f'Transfer complete: {received} bytes')
            self.get_logger().info('Unzipping ...')

            # ---> Step 3 <--- #
            
            self._unzip_file(zip_path, save_path)
            self.trigger_start_reading() # Trigger start reading after successful transfer and unzip

        # If connection error occurs, log and drop client
        except ConnectionError as e:
            self.get_logger().warn(str(e))
            self._drop_client()
        except Exception as e:
            self.get_logger().error(f'Receive error: {e}')
            self._drop_client()
        finally:
            # ---> Step 4 <--- #

            # Clean up the zip file
            if os.path.exists(zip_path):
                os.remove(zip_path)
                self.get_logger().info('Cleaned up temporary zip file.')
            #Maybe drop client after transfer
        
    def _unzip_file(self, zip_path: str, extract_to: str):
        """
        Unzips the rosbag zip file into specified directory

        zip_path: Path to the zip file
        extract_to: Directory to extract the contents to

        after unzipping the bag should be available in
        {extract_to}/scan_timestamp
        """

        self.get_logger().info(f'Unzipping to {extract_to} ...')

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extracts all the contents to the target directory
            zf.extractall(extract_to)
        
        extracted = os.listdir(extract_to)
        self.get_logger().info(f'Extracted: {extracted}')
        



    def _drop_client(self):
        with self._lock:
            self._connected = False
            try:
                self._client_sock.close()
            except Exception:
                pass
            self._client_sock = None

    def _cb_disconnect(self, _request, response):
        if not self._connected:
            response.success = False
            response.message = 'No client connected.'
        else:
            self._drop_client()
            response.success = True
            response.message = 'Client disconnected.'
        return response

    def destroy_node(self):
        self._drop_client()
        try:
            self._server_sock.close()
        except Exception:
            pass
        super().destroy_node()

    def trigger_start_reading(self):
        request = Trigger.Request()
        future = self.start_reading_client.call_async(request)
        future.add_done_callback(self.start_reading_response_cb)
    
    def start_reading_response_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info(f'Start reading: {result.message}')
        else:
            self.get_logger().error(f'Failed to start reading: {result.message}')


def main(args=None):
    rclpy.init(args=args)
    node = UGV_SvrSide()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()