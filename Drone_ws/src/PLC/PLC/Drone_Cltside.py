#!/usr/bin/env python3

import socket
import struct
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger


class Drone_Cltside(Node):
    """
    ROS 2 TCP client node.

    1. Connects to the TcpMeanServerNode on startup.
    2. Receives a numpy array frame from the server.
    3. Computes the mean of the array.
    4. Sends the mean back as a 4-byte float32 LE.
    5. Publishes the received array and computed mean to ROS topics.

    Wire protocol - server to client (numpy frame):
        [8 bytes] magic   b'NPARR\x00\x00\x00'
        [1 byte]  dtype   e.g. b'f' = float32
        [1 byte]  ndim    uint8
        [ndim x 4 bytes]  shape: uint32 LE per dim
        [N bytes] data    raw bytes

    Wire protocol - client to server (mean reply):
        [4 bytes] float32 LE

    Parameters:
        host (str) - server IP address  [default: '127.0.0.1']
        port (int) - server port        [default: 12347]

    Topics published:
        ~/array (std_msgs/Float32MultiArray) - array received from server
        ~/mean  (std_msgs/Float32)           - computed mean sent back

    Services:
        ~/connect    (std_srvs/Trigger) - (re)connect to the server
        ~/disconnect (std_srvs/Trigger) - disconnect from the server
    """

    MAGIC     = b'NPARR\x00\x00\x00'
    MAGIC_LEN = 8
    DTYPE_MAP = {
        b'f': np.float32, b'd': np.float64,
        b'i': np.int32,   b'l': np.int64,
        b'B': np.uint8,
    }

    def __init__(self):
        super().__init__('drone_Client_side')

        # parameters
        self.declare_parameter('host', '10.42.0.1')
        self.declare_parameter('port', 12347)

        # state
        self._sock: socket.socket | None = None
        self._connected = False
        self._lock = threading.Lock()

        # publishers
        self._array_pub = self.create_publisher(Float32MultiArray, '~/array', 10)
        self._mean_pub  = self.create_publisher(Float32,           '~/mean',  10)

        # services
        self.create_service(Trigger, '~/connect',    self._cb_connect)
        self.create_service(Trigger, '~/disconnect', self._cb_disconnect)

        # connect on startup
        self._connect()

    # connection helpers

    def _connect(self):
        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value

        if self._connected:
            return False, 'Already connected.'

        try:
            sock = socket.socket() # socket.AF_INET, socket.SOCK_STREAM
            sock.connect((host, port))
            with self._lock:
                self._sock      = sock
                self._connected = True
            self.get_logger().info(f'Connected to {host}:{port}')
            threading.Thread(target=self._recv_loop, daemon=True).start()
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

    # receive loop

    def _recv_exactly(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError('Server closed the connection.')
            buf += chunk
        return buf

    def _read_frame(self) -> np.ndarray:
        magic = self._recv_exactly(self.MAGIC_LEN)
        if magic != self.MAGIC:
            raise ValueError(f'Bad magic: {magic!r}')

        dtype_char = self._recv_exactly(1)
        dtype = self.DTYPE_MAP.get(dtype_char)
        if dtype is None:
            raise ValueError(f'Unknown dtype: {dtype_char!r}')

        ndim  = struct.unpack('B', self._recv_exactly(1))[0]
        shape = struct.unpack(f'<{ndim}I', self._recv_exactly(ndim * 4))

        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        raw     = self._recv_exactly(n_bytes)

        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def _recv_loop(self):
        while self._connected:
            try:
                array = self._read_frame()
                mean  = float(array.mean())

                # Send mean back
                with self._lock:
                    if self._connected and self._sock:
                        self._sock.sendall(struct.pack('<f', mean))

                self.get_logger().info(
                    f'Received array  shape={array.shape}  '
                    f'dtype={array.dtype}  mean={mean:.6f}  -> sent mean back'
                )
                self._publish_array(array)
                self._publish_mean(mean)

            except ConnectionError as e:
                self.get_logger().warn(str(e))
                self._connected = False
                break
            except Exception as e:
                if self._connected:
                    self.get_logger().error(f'Receive error: {e}')
                self._connected = False
                break

    # publish helpers

    def _publish_array(self, array: np.ndarray):
        msg = Float32MultiArray()
        for i, size in enumerate(array.shape):
            dim        = MultiArrayDimension()
            dim.label  = f'dim{i}'
            dim.size   = size
            dim.stride = int(np.prod(array.shape[i:]))
            msg.layout.dim.append(dim)
        msg.data = array.astype(np.float32).flatten().tolist()
        self._array_pub.publish(msg)

    def _publish_mean(self, mean: float):
        msg      = Float32()
        msg.data = mean
        self._mean_pub.publish(msg)

    # service callbacks

    def _cb_connect(self, _request, response):
        response.success, response.message = self._connect()
        return response

    def _cb_disconnect(self, _request, response):
        response.success, response.message = self._disconnect()
        return response

    def destroy_node(self):
        self._disconnect()
        super().destroy_node()


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