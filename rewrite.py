import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from nav_msgs.msg import Odometry, Path
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

def remove_outliers(points, threshold=0.1):
    """使用距离阈值剔除异常点"""
    if len(points) < 3:
        return points
    
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    median_dist = np.median(min_dists)
    
    keep = min_dists < threshold * median_dist
    return points[keep]

def smooth_path(x, y, gaussian_sigma=3.0, sg_window=150, sg_order=2):
    """多阶段平滑处理路径"""
    points = np.column_stack((x, y))
    
    if len(points) > 10:
        points = remove_outliers(points)
    
    if len(points) < 2:
        return x, y
    
    x_raw, y_raw = points[:, 0], points[:, 1]
    
    x_gauss = gaussian_filter1d(x_raw, sigma=gaussian_sigma, mode='nearest')
    y_gauss = gaussian_filter1d(y_raw, sigma=gaussian_sigma, mode='nearest')
    
    window_size = min(sg_window, len(x_gauss) - 1) 
    if window_size % 2 == 0:
        window_size = max(3, window_size - 1)
    
    if window_size > sg_order:
        x_smooth = savgol_filter(x_gauss, window_length=window_size, polyorder=sg_order)
        y_smooth = savgol_filter(y_gauss, window_length=window_size, polyorder=sg_order)
    else:
        x_smooth, y_smooth = x_gauss, y_gauss
    
    return x_smooth, y_smooth

def process_and_write_bag(input_path, output_path):
    # 初始化读写器
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=input_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)
    
    writer = rosbag2_py.SequentialWriter()
    writer_storage_options = rosbag2_py.StorageOptions(uri=output_path, storage_id="sqlite3")
    writer.open(writer_storage_options, converter_options)
    
    # 获取所有话题并注册
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    
    for topic in topic_types:
        writer.create_topic(topic)
    
    # 初始化坐标列表
    x_coords, y_coords = [], []
    odom_msgs = []
    timestamps = []
    
    # 第一遍读取：收集数据用于平滑
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        
        if topic == "/odom_info":
            x_coords.append(msg.pose.pose.position.x)
            y_coords.append(msg.pose.pose.position.y)
            odom_msgs.append(msg)
            timestamps.append(timestamp)
        elif topic == "/path":
            for pose in msg.poses:
                x_coords.append(pose.pose.position.x)
                y_coords.append(pose.pose.position.y)
    
    # 平滑处理
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x_smooth, y_smooth = smooth_path(x_coords, y_coords)
    
    # 第二遍读取：写入处理后的数据
    reader = rosbag2_py.SequentialReader()  # 重新初始化读取器
    reader.open(storage_options, converter_options)
    
    odom_idx = 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        
        if topic == "/odom_info":
            # 应用平滑后的坐标
            processed_msg = odom_msgs[odom_idx]
            processed_msg.pose.pose.position.x = x_smooth[odom_idx]
            processed_msg.pose.pose.position.y = y_smooth[odom_idx]
            odom_idx += 1
            
            writer.write(topic, serialize_message(processed_msg), timestamp)
        else:
            # 其他话题原样写入
            writer.write(topic, data, timestamp)

if __name__ == "__main__":
    input_bag = "/home/hefangyuan/python_projects/bagfiles_process/test/test.db3"
    output_bag = "/home/hefangyuan/python_projects/bagfiles_process/test/processed.db3"
    
    if not os.path.exists(input_bag):
        print('Bag file not found')
        exit()
    
    print('Processing bag file...')
    process_and_write_bag(input_bag, output_bag)
    print(f'Processing complete. Output saved to {output_bag}')