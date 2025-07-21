import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry, Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
import os

# 1. 读取 ROS Bag 数据
bag_path = "/home/hefangyuan/bag_files/bag3_arc/bag3.db3"
if not os.path.exists(bag_path):
    raise FileNotFoundError("Bag file not found!")

x_coords, y_coords = [], []

# 打开 rosbag2 文件
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
reader.open(storage_options, rosbag2_py.ConverterOptions())

# 遍历消息
while reader.has_next():
    topic, data, timestamp = reader.read_next()
    if topic == "/odom_info":
        msg = deserialize_message(data, Odometry)
        x_coords.append(msg.pose.pose.position.x)
        y_coords.append(msg.pose.pose.position.y)
    elif topic == "/path":
        msg = deserialize_message(data, Path)
        for pose in msg.poses:
            x_coords.append(pose.pose.position.x)
            y_coords.append(pose.pose.position.y)

x, y = np.array(x_coords), np.array(y_coords)

# 2. 使用 Savitzky-Golay 滤波器平滑数据（适用于轨迹平滑）
window_length = min(51, len(x) // 2 * 2 + 1)  # 窗口大小（必须为奇数）
window_length = 10001
if window_length > 3:  # 至少需要 3 个点
    y_smooth_sg = savgol_filter(y, window_length=window_length, polyorder=2)
else:
    y_smooth_sg = y

# 4. 绘制原始数据和平滑后的数据
plt.figure(figsize=(12, 6))

# 原始数据
plt.plot(x, y, 'r.', markersize=4, alpha=0.4, label="Original Data")

# Savitzky-Golay 平滑
plt.plot(x, y_smooth_sg, 'b-', linewidth=2, label="Savitzky-Golay Smoothed")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("ROS 2 Path Smoothing Comparison")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.tight_layout()
plt.show()