#!/usr/bin/env python3

import socket
import struct
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger


class UGV_SvrSide(Node):
    """
    ROS 2 TCP server node.

    1. Waits for one client to connect.
    2. Sends a random float32 array to the client on a timer.
    3. Reads the mean (4-byte float32) the client sends back.
    4. Publishes the received mean to ~/mean.

    Wire protocol - server to client (numpy frame):
        [8 bytes] magic   b'NPARR\x00\x00\x00'
        [1 byte]  dtype   b'f' = float32
        [1 byte]  ndim    uint8
        [ndim x 4 bytes]  shape: uint32 LE per dim
        [N bytes] data    raw bytes

    Wire protocol - client to server (mean reply):
        [4 bytes] float32 LE

    Parameters:
        host          (str)   - bind address          [default: '0.0.0.0']
        port          (int)   - bind port              [default: 12347]
        send_interval (float) - seconds between arrays [default: 1.0]
        array_rows    (int)   - rows of random array   [default: 4]
        array_cols    (int)   - cols of random array   [default: 4]

    Topics published:
        ~/sent_array (std_msgs/Float32MultiArray) - array sent to client
        ~/mean       (std_msgs/Float32)           - mean received from client

    Services:
        ~/disconnect (std_srvs/Trigger) - drop the current client
    """

    MAGIC      = b'NPARR\x00\x00\x00'
    DTYPE_CHAR = {
        np.float32: b'f', np.float64: b'd',
        np.int32:   b'i', np.int64:   b'l',
        np.uint8:   b'B',
    }

    def __init__(self):
        super().__init__('ugv_server_side')

        # parameters
        self.declare_parameter('host',          '0.0.0.0')
        self.declare_parameter('port',          12347)
        self.declare_parameter('send_interval', 1.0)
        self.declare_parameter('array_rows',    4)
        self.declare_parameter('array_cols',    4)

        # state
        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._connected  = False
        self._lock       = threading.Lock()
        self._send_timer = None

        # publishers
        self._array_pub = self.create_publisher(Float32MultiArray, '~/sent_array', 10)
        self._mean_pub  = self.create_publisher(Float32,           '~/mean',       10)

        # service
        self.create_service(Trigger, '~/disconnect', self._cb_disconnect)

        # start listening in background
        threading.Thread(target=self._listen_loop, daemon=True).start()

    # listen / accept loop

    def _listen_loop(self):
        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((host, port))
        self._server_sock.listen(1)
        self.get_logger().info(f'Listening on {host}:{port} ...')

        while rclpy.ok():
            try:
                conn, addr = self._server_sock.accept()
            except Exception as e:
                if rclpy.ok():
                    self.get_logger().error(f'Accept error: {e}')
                break

            with self._lock:
                self._client_sock = conn
                self._connected   = True
            self.get_logger().info(f'Client connected: {addr}')

            # Create send timer once on first connection
            if self._send_timer is None:
                interval = (self.get_parameter('send_interval')
                            .get_parameter_value().double_value)
                self._send_timer = self.create_timer(interval, self._cb_send_array)

            # Block this thread reading mean replies until client disconnects
            self._recv_mean_loop()
            self.get_logger().info('Waiting for next client ...')

    # receive mean loop (runs in listen thread)

    def _recv_exactly(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self._client_sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError('Client closed the connection.')
            buf += chunk
        return buf

    def _recv_mean_loop(self):
        while self._connected:
            try:
                raw  = self._recv_exactly(4)
                mean = struct.unpack('<f', raw)[0]
                self.get_logger().info(f'Received mean from client: {mean:.6f}')

                msg      = Float32()
                msg.data = float(mean)
                self._mean_pub.publish(msg)

            except ConnectionError as e:
                self.get_logger().warn(str(e))
                self._drop_client()
                break
            except Exception as e:
                if self._connected:
                    self.get_logger().error(f'Mean recv error: {e}')
                self._drop_client()
                break

    # send array (ROS timer callback)

    def _build_frame(self, array: np.ndarray) -> bytes:
        dtype_char = self.DTYPE_CHAR[array.dtype.type]
        header = (
            self.MAGIC
            + dtype_char
            + struct.pack('B', array.ndim)
            + struct.pack(f'<{array.ndim}I', *array.shape)
        )
        return header + np.ascontiguousarray(array).tobytes()

    def _cb_send_array(self):
        rows = self.get_parameter('array_rows').get_parameter_value().integer_value
        cols = self.get_parameter('array_cols').get_parameter_value().integer_value

        with self._lock:
            if not self._connected:
                return
            try:
                arr   = np.random.rand(rows, cols).astype(np.float32)
                frame = self._build_frame(arr)
                self._client_sock.sendall(frame)
            except Exception as e:
                self.get_logger().error(f'Send error: {e}')
                self._connected = False
                return

        # Publish sent array
        ros_msg = Float32MultiArray()
        for i, size in enumerate(arr.shape):
            dim        = MultiArrayDimension()
            dim.label  = f'dim{i}'
            dim.size   = size
            dim.stride = int(np.prod(arr.shape[i:]))
            ros_msg.layout.dim.append(dim)
        ros_msg.data = arr.flatten().tolist()
        self._array_pub.publish(ros_msg)
        self.get_logger().info(
            f'Sent array  shape={arr.shape}  true_mean={arr.mean():.6f}'
        )

    # helpers

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