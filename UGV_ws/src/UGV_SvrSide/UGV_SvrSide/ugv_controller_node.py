import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
import time
import threading
import serial
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def BSpline(points, smoothing, k):
    points = np.array(points)
    if len(points) < k + 1:
        raise ValueError(f"Need at least {k+1} points for a degree-{k} spline, got {len(points)}")

    # Chord-length parameterisation
    d = 0
    t = [0]
    for i in range(len(points) - 1):
        d += np.sqrt(np.sum((points[i+1] - points[i])**2))
        t.append(d)
    t = np.array(t)
    t = t / t[-1]   # normalise 0→1

    tck, _ = splprep([points[:, 0], points[:, 1]], u=t, s=smoothing, k=k)
    total_length = t[-1]
    n_points = int(np.clip(total_length * 10, 50, 5000))
    u_fine  = np.linspace(0, 1, n_points)

    x_u,  y_u  = splev(u_fine, tck)         # positions
    dx_u, dy_u = splev(u_fine, tck, der=1)  # first derivative
    ddx_u, ddy_u = splev(u_fine, tck, der=2)  # second derivative

    speed = np.sqrt(dx_u**2 + dy_u**2)
    kp = (dx_u * ddy_u - dy_u * ddx_u) / (speed**3 + 1e-9)  # Curvature for General Parametrizations.
    v_max = np.sqrt(1.0 / (np.abs(kp) + 1e-6))              # Max speed based on curvature. 
    v_max = np.clip(v_max, 0, 80)

    return v_max, x_u, y_u

class UGVController(Node):
    def __init__(self):
        super().__init__('ugv_controller')

        self.declare_parameter('input_source', 'serial')
        self.source = self.get_parameter('input_source').value
        self.get_logger().info(f"Using input source: {self.source}")
        self.declare_parameter('test_mode', True)
        self.test_mode = self.get_parameter('test_mode').value

        self.ser = None
        if not self.test_mode:
            self.ser = serial.Serial('/dev/ttyUSB1', baudrate=115200, timeout=1.0)

        # Subscribers
        self.create_subscription(String, '/ugv/state', self.state_callback, 10)

        if self.source == 'ros':
            self.create_subscription(
                PoseStamped, 'ugv/position', self.position_callback, 10)

        self.create_subscription(String, 'ugv/astar_points',   self.astar_points_callback, 10)
        self.create_subscription(String, 'uav/radio_in/mission_command', self.mission_command_callback, 10)

        # Publishers
        self.steering_angle_pub = self.create_publisher(Float32, '/ugv/steering_angle', 10)

        # State
        self.current_x     = 0.0   # metres, world frame
        self.current_y     = 0.0
        self.current_z     = 0.0
        self.current_angle = 0.0   # degrees

        self.target_x = 0.0
        self.target_y = 0.0

        self.astar_points = []
        self.spline_ready = False
        self.s       = None
        self.x_smooth = None
        self.y_smooth = None

        self.L = 0.480          # wheelbase
        self.lookahead = 0.5    # lookahead distance
        self.metres_per_count = 0.57203 / 100 # Wheel circumference / counts per revolution

        self.RATE_HZ = 20
        self.dt = 1.0 / self.RATE_HZ
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Serial thread
        self.zero_found = False
        self.last_steer_deg = 90   # 90 = straight ahead after the +90 offset
        self.last_sent = 0
        if not self.test_mode:
            self.serial_thread = threading.Thread(target=self.serial_loop, daemon=True)
            self.serial_thread.start()

        if self.test_mode:
            self.load_test_path()
            self.setup_plot()

        self.get_logger().info('ugv_controller node started')

    # Callbacks
    def state_callback(self, msg):
        self.get_logger().info(f"UGV state: {msg.data}")

    def mission_command_callback(self, msg):
        self.get_logger().info(f"Mission command received: {msg.data}")

    def position_callback(self, msg): # Used when input_source == 'ros'.
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z

        # Convert quaternion to yaw
        r = R.from_quat([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        yaw_deg = r.as_euler('zxy', degrees=True)[0]
        self.current_angle = yaw_deg

        if self.spline_ready:
            self.target_x, self.target_y = self.target_point(self.x_smooth, self.y_smooth)

    def astar_points_callback(self, msg): # expected format: "[[x0,y0],[x1,y1],...]"
        try:
            parsed = json.loads(msg.data)
            self.astar_points = [list(p) for p in parsed]
        except (json.JSONDecodeError, ValueError) as e:
            self.get_logger().error(f"Bad astar_points message: {e}")
            return

        if len(self.astar_points) < 6:
            self.get_logger().warn("Too few waypoints for degree-5 spline; need ≥ 6.")
            return

        try:
            points_np = np.array(self.astar_points, dtype=float)
            self.s, self.x_smooth, self.y_smooth = BSpline(points_np, 20000, 5)
            self.spline_ready = True
            self.target_x, self.target_y = self.target_point(self.x_smooth, self.y_smooth)
            self.get_logger().info(f"Spline fitted to {len(self.astar_points)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"BSpline fitting failed: {e}")


    def target_point(self, x_smooth, y_smooth): # Walk lookahead distance along the spline from the closest point.
        # Find the index of the spline point closest to the car
        dists = np.sqrt((x_smooth - self.current_x)**2 + (y_smooth - self.current_y)**2) # list
        u0 = int(np.argmin(dists))

        # Walk forward along the spline until we've covered tDistance
        s = 0.0
        uL = u0
        for i in range(u0, len(x_smooth) - 1):
            ds = np.sqrt((x_smooth[i+1] - x_smooth[i])**2 + (y_smooth[i+1] - y_smooth[i])**2)
            s += ds
            if s >= self.lookahead:
                uL = i
                break
        else:
            uL = len(x_smooth) - 1   # reached end of spline

        return float(x_smooth[uL]), float(y_smooth[uL])

    def steering_angle(self, target_x, target_y):
        dx = target_x - self.current_x
        dy = target_y - self.current_y

        # Target in world frame (radians)
        angle_world = np.arctan2(dy, dx)

        # Alpha: angular error between heading and bearing to target
        # Normalise to [-pi, pi]
        alpha = angle_world - np.radians(self.current_angle)
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        Ld = np.hypot(dx, dy)
        if Ld < 1e-6:
            return 0.0   # already at target — go straight

        # Bicycle model
        delta = np.arctan2(2.0 * self.L * np.sin(alpha), Ld)
        return delta

    # Serial thread
    def serial_loop(self): #Arduino sends lines like 'angle_deg,encoder_counts\n'
        while True:
            try:

                # Write
                if self.last_sent < time.time() - self.dt:
                    steer_byte = bytes([int(np.clip(self.last_steer_deg, 0, 180))])
                    self.ser.write(steer_byte)

                if self.source == 'ros':
                    continue

                # Read
                raw = self.ser.readline().decode(errors='ignore').strip()
                if raw and ',' in raw:
                    try:
                        angle_str, count_str = raw.split(',', 1)
                        angle_deg = float(angle_str)
                        count     = float(count_str)

                        if not self.zero_found: # Sanity check because the arduino sometimes sends wrong encoder counts before/during reset.
                            if count < 1:
                                self.zero_found = True
                        else:
                            self.current_angle = angle_deg
                            distance = count * self.metres_per_count
                            self.current_x += distance * np.cos(self.current_angle)
                            self.current_y += distance * np.sin(self.current_angle)

                            if self.spline_ready:
                                self.target_x, self.target_y = self.target_point(
                                    self.x_smooth, self.y_smooth)

                    except ValueError:
                        self.get_logger().warn(f"Malformed serial line: {raw}")

            except serial.SerialException as e:
                self.get_logger().error(f"Serial error: {e}")
                time.sleep(1.0)

            time.sleep(self.dt/2) # 40 Hz

    def setup_plot(self):

        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.raw_line, = self.ax.plot([], [], 'b-', markersize=2, label='A-star')
        self.path_line, = self.ax.plot([], [], 'r-', label='Trajectory')

        #self.vehicle_dot, = self.ax.plot([], [], 'ro')
        #self.target_dot, = self.ax.plot([], [], 'go')

        self.ax.legend()
        self.ax.set_aspect('equal')
        self.ax.tick_params(axis='both', labelsize=14)
        self.ax.set_xlabel("X position", fontsize=16)
        self.ax.set_ylabel("Y position", fontsize=16)
        self.anim = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,
            cache_frame_data=False
        )
        plt.show()

    def update_plot(self, frame):

        if self.spline_ready:

            self.path_line.set_data(
                -self.x_smooth,
                self.y_smooth
            )

            raw = np.array(self.astar_points)

            self.raw_line.set_data(
                -raw[:, 0],
                raw[:, 1]
            )

            # self.vehicle_dot.set_data(
            #     [self.current_x],
            #     [self.current_y]
            # )

            # self.target_dot.set_data(
            #     [self.target_x],
            #     [self.target_y]
            # )

            self.ax.relim()
            self.ax.autoscale_view()

            self.fig.canvas.draw_idle()

        return (
            self.path_line,
            self.raw_line,
            #self.vehicle_dot,
            #self.target_dot
        )

    def load_test_path(self):

        # Example path similar to your data
        #points = []

        # Horizontal segment
        # for x in range(19, 35):
        #     points.append((x, 69))

        # # Diagonal segment
        # for i in range(35, 490):
        #     points.append((i, i + 35))

        # # Vertical-ish tail
        # y = 645
        # for i in range(490, 515):
        #     points.append((490 + (i - 490), y + (i - 490)))
        points1 = [[4.8500000000000005, 16.75], [4.8500000000000005, 16.7], [4.8500000000000005, 16.650000000000002], [4.8500000000000005, 16.6], [4.8500000000000005, 16.55], [4.8500000000000005, 16.5], [4.8500000000000005, 16.45], [4.8500000000000005, 16.400000000000002], [4.8500000000000005, 16.35], [4.8500000000000005, 16.3], [4.8500000000000005, 16.25], [4.8500000000000005, 16.2], [4.8500000000000005, 16.150000000000002], [4.8500000000000005, 16.1], [4.8500000000000005, 16.05], [4.8500000000000005, 16.0], [4.8500000000000005, 15.950000000000001], [4.8500000000000005, 15.9], [4.8500000000000005, 15.850000000000001], [4.8500000000000005, 15.8], [4.8500000000000005, 15.75], [4.8500000000000005, 15.700000000000001], [4.8500000000000005, 15.65], [4.8500000000000005, 15.600000000000001], [4.8500000000000005, 15.55], [4.8500000000000005, 15.5], [4.8500000000000005, 15.450000000000001], [4.8500000000000005, 15.4], [4.8500000000000005, 15.350000000000001], [4.8500000000000005, 15.3], [4.800000000000001, 15.25], [4.75, 15.200000000000001], [4.7, 15.15], [4.65, 15.100000000000001], [4.6000000000000005, 15.05], [4.55, 15.0], [4.5, 14.950000000000001], [4.45, 14.9], [4.4, 14.850000000000001], [4.3500000000000005, 14.8], [4.3, 14.75], [4.25, 14.700000000000001], [4.2, 14.65], [4.15, 14.600000000000001], [4.1000000000000005, 14.55], [4.05, 14.5], [4.0, 14.450000000000001], [4.0, 14.4], [4.0, 14.350000000000001], [4.0, 14.3], [4.0, 14.25], [4.0, 14.200000000000001], [4.0, 14.15], [4.0, 14.100000000000001], [4.0, 14.05], [4.0, 14.0], [4.0, 13.950000000000001], [4.0, 13.9], [4.0, 13.850000000000001], [4.0, 13.8], [4.0, 13.75], [4.0, 13.700000000000001], [4.0, 13.65], [4.0, 13.600000000000001], [4.0, 13.55], [4.0, 13.5], [4.0, 13.450000000000001], [4.0, 13.4], [4.0, 13.350000000000001], [4.0, 13.3], [4.05, 13.25], [4.1000000000000005, 13.200000000000001], [4.15, 13.15], [4.2, 13.100000000000001], [4.25, 13.05], [4.3, 13.0], [4.3500000000000005, 12.950000000000001], [4.4, 12.9], [4.45, 12.850000000000001], [4.45, 12.8], [4.45, 12.75], [4.5, 12.700000000000001], [4.55, 12.65], [4.6000000000000005, 12.600000000000001], [4.65, 12.55], [4.7, 12.5], [4.75, 12.450000000000001], [4.800000000000001, 12.4], [4.8500000000000005, 12.350000000000001], [4.8500000000000005, 12.3], [4.8500000000000005, 12.25], [4.8500000000000005, 12.200000000000001], [4.8500000000000005, 12.15], [4.8500000000000005, 12.100000000000001], [4.8500000000000005, 12.05], [4.8500000000000005, 12.0], [4.8500000000000005, 11.950000000000001], [4.8500000000000005, 11.9], [4.8500000000000005, 11.850000000000001], [4.8500000000000005, 11.8], [4.8500000000000005, 11.75]]
        points = [(19, 69), (20, 69), (21, 69), (22, 69), (23, 69), (24, 69), (25, 69), 
                (26, 69), (27, 69), (28, 69), (29, 69), (30, 69), (31, 69), (32, 69), 
                (33, 69), (34, 69), (35, 70), (36, 71), (37, 72), (38, 73), (39, 74), 
                (40, 75), (41, 76), (42, 77), (43, 78), (44, 79), (45, 80), (46, 81), 
                (47, 82), (48, 83), (49, 84), (50, 85), (51, 86), (52, 87), (53, 88), 
                (54, 89), (55, 90), (56, 91), (57, 92), (58, 93), (59, 94), (60, 95), 
                (61, 96), (62, 97), (63, 98), (64, 99), (65, 100), (66, 101), (67, 102), 
                (68, 103), (69, 104), (70, 105), (71, 106), (72, 107), (73, 108), (74, 109), 
                (75, 110), (76, 111), (77, 112), (78, 113), (79, 114), (80, 115), (81, 116), 
                (82, 117), (83, 118), (84, 119), (85, 120), (86, 121), (87, 122), (88, 123), 
                (89, 124), (90, 125), (91, 126), (92, 127), (93, 128), (94, 129), (95, 130), 
                (96, 131), (97, 131), (98, 131), (99, 131), (100, 131), (101, 131), (102, 131), 
                (103, 131), (104, 131), (105, 131), (106, 131), (107, 131), (108, 131), (109, 131), 
                (110, 131), (111, 131), (112, 131), (113, 131), (114, 131), (115, 131), (116, 131), 
                (117, 131), (118, 131), (119, 131), (120, 131), (121, 131), (122, 131), (123, 131), 
                (124, 131), (125, 131), (126, 131), (127, 131), (128, 131), (129, 131), (130, 131), 
                (131, 131), (132, 131), (133, 131), (134, 131), (135, 131), (136, 131), (137, 131), 
                (138, 131), (139, 131), (140, 131), (141, 131), (142, 131), (143, 131), (144, 131), 
                (145, 131), (146, 131), (147, 131), (148, 131), (149, 131), (150, 131), (151, 131), 
                (152, 131), (153, 131), (154, 131), (155, 131), (156, 131), (157, 131), (158, 131), 
                (159, 131), (160, 131), (161, 131), (162, 131), (163, 131), (164, 131), (165, 131), 
                (166, 131), (167, 131), (168, 131), (169, 131), (170, 131), (171, 131), (172, 131), 
                (173, 131), (174, 131), (175, 131), (176, 131), (177, 131), (178, 131), (179, 131), 
                (180, 131), (181, 131), (182, 131), (183, 131), (184, 131), (185, 131), (186, 131), 
                (187, 131), (188, 131), (189, 131), (190, 131), (191, 131), (192, 131), (193, 131), 
                (194, 131), (195, 131), (196, 131), (197, 131), (198, 131), (199, 131), (200, 131), 
                (201, 131), (202, 131), (203, 131), (204, 131), (205, 131), (206, 131), (207, 131), 
                (208, 131), (209, 131), (210, 131), (211, 131), (212, 131), (213, 131), (214, 131), 
                (215, 131), (216, 131), (217, 131), (218, 131), (219, 131), (220, 131), (221, 131), 
                (222, 132), (223, 133), (224, 134), (225, 135), (226, 136), (227, 137), (228, 138), 
                (229, 139), (230, 140), (231, 141), (232, 142), (233, 143), (234, 144), (235, 145), 
                (236, 146), (237, 147), (238, 148), (239, 149), (240, 150), (241, 151), (241, 152), 
                (241, 153), (241, 154), (241, 155), (242, 156), (243, 157), (244, 158), (245, 159), 
                (246, 160), (247, 161), (248, 162), (249, 163), (250, 164), (251, 165), (251, 166), 
                (251, 167), (251, 168), (251, 169), (251, 170), (251, 171), (251, 172), (252, 173), 
                (253, 174), (254, 175), (255, 176), (256, 177), (257, 178), (258, 179), (259, 180), 
                (260, 181), (261, 182), (261, 183), (261, 184), (261, 185), (261, 186), (261, 187), 
                (261, 188), (261, 189), (261, 190), (261, 191), (261, 192), (261, 193), (261, 194), 
                (261, 195), (261, 196), (262, 197), (263, 198), (264, 199), (265, 200), (265, 201), 
                (265, 202), (265, 203), (265, 204), (265, 205), (265, 206), (265, 207), (265, 208), 
                (265, 209), (265, 210), (265, 211), (265, 212), (265, 213), (265, 214), (266, 215), 
                (267, 216), (268, 217), (269, 218), (269, 219), (269, 220), (269, 221), (269, 222), 
                (269, 223), (269, 224), (269, 225), (269, 226), (270, 227), (271, 228), (272, 229), 
                (273, 230), (274, 231), (275, 232), (276, 233), (277, 234), (278, 235), (279, 236), 
                (279, 237), (279, 238), (279, 239), (279, 240), (279, 241), (279, 242), (279, 243), 
                (279, 244), (279, 245), (279, 246), (279, 247), (279, 248), (279, 249), (279, 250), 
                (279, 251), (279, 252), (279, 253), (279, 254), (279, 255), (279, 256), (280, 257), 
                (280, 258), (281, 259), (281, 260), (281, 261), (281, 262), (281, 263), (281, 264), 
                (281, 265), (281, 266), (281, 267), (281, 268), (281, 269), (281, 270), (281, 271), 
                (281, 272), (281, 273), (281, 274), (281, 275), (281, 276), (282, 277), (283, 278), 
                (284, 279), (285, 280), (286, 281), (287, 282), (288, 283), (289, 284), (290, 285), 
                (291, 286), (292, 287), (293, 288), (294, 289), (295, 290), (296, 291), (297, 292), 
                (298, 293), (299, 294), (300, 295), (301, 296), (302, 297), (303, 298), (304, 299), 
                (305, 300), (306, 301), (307, 302), (308, 303), (309, 304), (310, 305), (311, 306), 
                (312, 307), (313, 308), (314, 309), (315, 310), (316, 311), (317, 312), (318, 313), 
                (319, 314), (320, 315), (321, 316), (322, 317), (323, 318), (324, 319), (325, 320), 
                (326, 321), (327, 322), (328, 323), (329, 324), (330, 325), (331, 326), (332, 327), 
                (333, 328), (334, 329), (335, 330), (336, 331), (337, 332), (338, 333), (339, 334), 
                (340, 335), (341, 336), (342, 337), (343, 338), (344, 339), (345, 340), (346, 341), 
                (347, 342), (348, 343), (349, 344), (350, 345), (351, 346), (352, 347), (353, 348), 
                (354, 349), (355, 350), (356, 351), (357, 352), (358, 353), (359, 354), (360, 355), 
                (361, 356), (362, 357), (363, 358), (364, 359), (365, 360), (366, 361), (367, 362), 
                (368, 363), (369, 364), (370, 365), (371, 366), (372, 367), (373, 368), (374, 369), 
                (375, 370), (376, 371), (377, 372), (378, 373), (379, 374), (380, 375), (381, 376), 
                (382, 377), (383, 378), (384, 379), (385, 380), (386, 381), (387, 382), (388, 383), 
                (389, 384), (390, 385), (391, 386), (392, 387), (393, 388), (394, 389), (395, 390), 
                (396, 391), (397, 392), (398, 393), (399, 394), (400, 395), (401, 396), (402, 397), 
                (403, 398), (404, 399), (405, 400), (406, 401), (407, 402), (408, 403), (409, 404), 
                (410, 405), (411, 406), (412, 407), (413, 408), (414, 409), (414, 410), (414, 411), 
                (414, 412), (414, 413), (414, 414), (414, 415), (414, 416), (414, 417), (414, 418), 
                (414, 419), (414, 420), (414, 421), (415, 422), (416, 423), (417, 424), (418, 425), 
                (419, 426), (420, 427), (421, 428), (422, 429), (423, 430), (424, 431), (425, 432), 
                (426, 433), (427, 434), (428, 435), (429, 436), (430, 437), (431, 438), (432, 439), 
                (433, 440), (434, 441), (435, 442), (436, 443), (437, 444), (438, 445), (439, 446), 
                (440, 447), (441, 448), (442, 449), (443, 450), (444, 451), (445, 452), (446, 453), 
                (447, 454), (448, 455), (449, 456), (450, 457), (451, 458), (452, 459), (453, 460), 
                (454, 461), (455, 462), (456, 463), (457, 464), (458, 465), (459, 466), (460, 467), 
                (461, 468), (462, 469), (463, 470), (464, 471), (465, 472), (466, 473), (467, 474), 
                (468, 475), (469, 476), (470, 477), (471, 478), (471, 479), (471, 480), (471, 481), 
                (471, 482), (471, 483), (471, 484), (471, 485), (471, 486), (471, 487), (471, 488), 
                (471, 489), (471, 490), (471, 491), (471, 492), (471, 493), (471, 494), (471, 495), 
                (471, 496), (471, 497), (471, 498), (471, 499), (471, 500), (471, 501), (471, 502), 
                (471, 503), (471, 504), (471, 505), (471, 506), (471, 507), (471, 508), (471, 509), 
                (471, 510), (471, 511), (471, 512), (471, 513), (471, 514), (471, 515), (471, 516), 
                (471, 517), (471, 518), (471, 519), (471, 520), (471, 521), (471, 522), (471, 523), 
                (471, 524), (471, 525), (471, 526), (471, 527), (471, 528), (471, 529), (471, 530), 
                (471, 531), (471, 532), (471, 533), (471, 534), (471, 535), (471, 536), (471, 537), 
                (471, 538), (471, 539), (471, 540), (471, 541), (471, 542), (471, 543), (471, 544), 
                (471, 545), (471, 546), (471, 547), (471, 548), (471, 549), (471, 550), (471, 551), 
                (471, 552), (471, 553), (471, 554), (471, 555), (471, 556), (471, 557), (471, 558), 
                (471, 559), (471, 560), (471, 561), (471, 562), (471, 563), (471, 564), (471, 565), 
                (471, 566), (471, 567), (471, 568), (471, 569), (471, 570), (471, 571), (471, 572), 
                (471, 573), (471, 574), (471, 575), (471, 576), (471, 577), (471, 578), (471, 579), 
                (471, 580), (471, 581), (471, 582), (471, 583), (471, 584), (471, 585), (471, 586), 
                (471, 587), (471, 588), (471, 589), (471, 590), (471, 591), (471, 592), (472, 593), 
                (473, 594), (474, 595), (475, 596), (476, 597), (477, 598), (478, 599), (479, 600), 
                (480, 601), (481, 602), (482, 603), (483, 604), (484, 605), (485, 606), (486, 607), 
                (487, 608), (488, 609), (489, 610), (490, 611), (490, 612), (490, 613), (490, 614), 
                (490, 615), (490, 616), (490, 617), (490, 618), (490, 619), (490, 620), (490, 621), 
                (490, 622), (490, 623), (490, 624), (490, 625), (490, 626), (490, 627), (490, 628), 
                (490, 629), (490, 630), (490, 631), (490, 632), (490, 633), (490, 634), (490, 635), 
                (490, 636), (490, 637), (490, 638), (490, 639), (490, 640), (490, 641), (490, 642), 
                (490, 643), (490, 644), (490, 645), (490, 646), (490, 647), (490, 648), (490, 649), 
                (490, 650), (490, 651), (490, 652), (490, 653), (490, 654), (490, 655), (490, 656), 
                (490, 657), (490, 658), (490, 659), (491, 660), (492, 661), (493, 662), (494, 663), 
                (495, 664), (496, 665), (497, 666), (498, 667), (499, 668), (500, 669), (501, 670), 
                (502, 671), (503, 672), (504, 673), (505, 674), (506, 675), (507, 676), (508, 677), 
                (509, 678), (510, 679), (511, 680), (512, 681), (513, 682), (514, 683)]

        self.astar_points = points

        points_np = np.array(self.astar_points, dtype=float)

        self.s, self.x_smooth, self.y_smooth = BSpline(
            points_np,
            smoothing=1000,
            k=5
        )

        self.spline_ready = True

        # Start vehicle at beginning
        self.current_x = self.x_smooth[0]
        self.current_y = self.y_smooth[0]

        self.target_x, self.target_y = self.target_point(
            self.x_smooth,
            self.y_smooth
        )

        self.get_logger().info(
            f"Loaded test path with {len(points)} points"
        )

    # Control loop (20 Hz timer)
    def control_loop(self):
        if not self.spline_ready:
            return

        delta = self.steering_angle(self.target_x, self.target_y)

        # Convert to degrees and map to between 0 and 180 instead of -90 to 90.
        steer_deg = np.degrees(-delta)
        steer_deg = np.clip(steer_deg, -90, 90)
        steer_deg = steer_deg + 90

        self.last_steer_deg = int(steer_deg)
        msg = Float32()
        msg.data = float(steer_deg)
        self.steering_angle_pub.publish(msg)
        if self.test_mode:

            speed = 0.5  # m/s

            self.current_x += speed * self.dt * np.cos(
                np.radians(self.current_angle)
            )

            self.current_y += speed * self.dt * np.sin(
                np.radians(self.current_angle)
            )

            self.current_angle += np.degrees(delta) * self.dt * 2.0

            self.target_x, self.target_y = self.target_point(
                self.x_smooth,
                self.y_smooth
            )
        self.get_logger().info(
            f"Vehicle: ({self.current_x:.2f}, {self.current_y:.2f}) "
            f"Target: ({self.target_x:.2f}, {self.target_y:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = UGVController()

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )

    ros_thread.start()

    plt.show()

    node.destroy_node()
    rclpy.shutdown()