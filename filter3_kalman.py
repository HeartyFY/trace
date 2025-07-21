import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os
from pykalman import KalmanFilter

# 1. 读取 ROS Bag 数据
bag_path = "/home/hefangyuan/bag_files/bag3_arc/bag3.db3"
if not os.path.exists(bag_path):
    print("Bag file not found!")

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
x, y = np.array(x_coords), np.array(y_coords)

# 2. 使用 Savitzky-Golay 滤波器平滑数据
window = 10001

window_length_sg = min(window, len(x) // 2 * 2 + 1)  # 窗口大小（必须为奇数）
if window_length_sg > 3:  # 至少需要 3 个点
    y_smooth_sg = savgol_filter(y, window_length=window_length_sg, polyorder=3)
else:
    y_smooth_sg = y

window_length_sg = min(window, len(x) // 2 * 2 + 1)  # 窗口大小（必须为奇数）
if window_length_sg > 3:  # 至少需要 3 个点
    x_smooth_sg = savgol_filter(y, window_length=window_length_sg, polyorder=3)
else:
    x_smooth_sg = x
x_smooth_sg = x


def apply_kalman_filter(measurements):
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=np.eye(2),
        initial_state_mean=measurements[0],
        initial_state_covariance=10 * np.eye(2),
        observation_covariance=2 * np.eye(2),
        transition_covariance=5 * np.eye(2)
    )
    smoothed_means, _ = kf.smooth(measurements)
    return smoothed_means

measurements = np.column_stack((x_smooth_sg, y_smooth_sg))
x_kalman, y_kalman = apply_kalman_filter(measurements).T

# 4. 绘制结果对比
plt.figure(figsize=(15, 6))

# 原始数据
#plt.plot(x, y, 'r.', markersize=4, alpha=0.4, label="Original Data")

# Savitzky-Golay平滑
plt.plot(x, y_smooth_sg, 'b-', linewidth=2, label=f"Savitzky-Golay (window={window_length_sg})")

# 组合平滑(Savitzky-Golay + kalman)
plt.plot(x, y_kalman, 'g-', linewidth=2, label=f"Combined (SG + kalman)")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Path Smoothing: Savitzky-Golay with Gaussian Filter")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
"""
# 5. 分图显示不同平滑效果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 子图1: 原始数据 vs Savitzky-Golay

ax1.plot(x, y, 'r.', markersize=4, alpha=0.4, label="Original")
ax1.plot(x_smooth_sg, y_smooth_sg, 'b-', linewidth=2, label="Savitzky-Golay")
ax1.set_title("Savitzky-Golay Filter Only")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.axis('equal')

# 子图2: 原始数据 vs 组合平滑
#ax2.plot(x, y, 'r.', markersize=4, alpha=0.4, label="Original")
ax2.plot(x_kalman, y_kalman, 'g-', linewidth=2, label="Combined Filter")
ax2.set_title("Savitzky-Golay + Gaussian Filter")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.axis('equal')
"""
plt.tight_layout()
plt.show()