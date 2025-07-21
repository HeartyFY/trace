import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry, Path
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

def remove_outliers(points, threshold=10):
    """
    使用距离阈值剔除异常点
    :param points: 坐标点数组 (N,2)
    :param threshold: 距离阈值，超过此值的点将被视为异常值
    :return: 过滤后的点数组
    """
    if len(points) < 3:
        return points
    
    # 计算每个点到其邻居的平均距离
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)  # 忽略自身距离
    min_dists = np.min(dists, axis=1)
    median_dist = np.median(min_dists)
    
    # 保留距离小于阈值的点
    keep = min_dists < threshold * median_dist
    return points[keep]

def smooth_path(x, y, gaussian_sigma=3.0, sg_window=50, sg_order=2):
    """
    使用多阶段平滑处理路径:
    1. 异常值剔除
    2. 高斯滤波初步平滑
    3. Savitzky-Golay滤波精细平滑
    :param x: x坐标数组
    :param y: y坐标数组
    :param gaussian_sigma: 高斯核的标准差
    :param sg_window: Savitzky-Golay窗口大小(必须为奇数)
    :param sg_order: Savitzky-Golay多项式阶数
    :return: 平滑后的x,y坐标
    """
    # 将坐标转换为numpy数组
    points = np.column_stack((x, y))
    
    # 1. 异常值剔除
    if len(points) > 10:  # 只有点数足够时才进行异常值剔除
        points = remove_outliers(points)
    
    if len(points) < 2:
        return x, y
    
    # 分离x,y坐标
    x_raw, y_raw = points[:, 0], points[:, 1]
    
    # 2. 高斯滤波初步平滑
    x_gauss = gaussian_filter1d(x_raw, sigma=gaussian_sigma, mode='nearest')
    y_gauss = gaussian_filter1d(y_raw, sigma=gaussian_sigma, mode='nearest')
    
    # 3. Savitzky-Golay滤波精细平滑
    # 确保窗口大小是奇数且小于数据长度
    window_size = min(sg_window, len(x_gauss) - 1) 
    if window_size % 2 == 0:  # 确保是奇数
        window_size = max(3, window_size - 1)
    
    if window_size > sg_order:
        x_smooth = savgol_filter(x_gauss, window_length=window_size, polyorder=sg_order)
        y_smooth = savgol_filter(y_gauss, window_length=window_size, polyorder=sg_order)
    else:
        x_smooth, y_smooth = x_gauss, y_gauss
    
    return x_smooth, y_smooth



bag_path = "/home/hefangyuan/bag_files/trace/3/3.db3" 

if not os.path.exists(bag_path):
    print('bag file not found')
    exit()
else:
    print('bag file found, processing...')

# 初始化坐标列表
x_coords, y_coords = [], []

# 打开 rosbag2 文件
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
reader.open(storage_options, rosbag2_py.ConverterOptions())

# 遍历消息
while reader.has_next():
    topic, data, timestamp = reader.read_next()
    if topic == "/Odometry":  # 替换为你的话题名
        msg = deserialize_message(data, Odometry)
        x_coords.append(msg.pose.pose.position.x)
        y_coords.append(msg.pose.pose.position.y)
    elif topic == "/path":  # 如果是路径数据
        msg = deserialize_message(data, Path)
        for pose in msg.poses:
            x_coords.append(pose.pose.position.x)
            y_coords.append(pose.pose.position.y)

# 转换为numpy数组便于处理
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

# 平滑处理
x_smooth, y_smooth = smooth_path(x_coords, y_coords)

# 绘制结果比较
plt.figure(figsize=(15, 5))

# 原始路径
plt.subplot(1, 3, 1)
plt.plot(x_coords, y_coords, 'b-', alpha=0.7, label="Original Path")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Original Path")
plt.legend()
plt.grid(True)
plt.axis('equal')

# 仅高斯滤波
x_gauss = gaussian_filter1d(x_coords, sigma=1.0)
y_gauss = gaussian_filter1d(y_coords, sigma=1.0)
plt.subplot(1, 3, 2)
plt.plot(x_gauss, y_gauss, 'g-', alpha=0.7, label="Gaussian Only")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Gaussian Filter")
plt.legend()
plt.grid(True)
plt.axis('equal')

# 组合平滑结果
plt.subplot(1, 3, 3)
plt.plot(x_smooth, y_smooth, 'r-', label="Combined Smoothing")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Gauss + Outlier Removal + Savitzky-Golay")
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()