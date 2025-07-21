import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry, Path
import matplotlib.pyplot as plt
import os

bag_path = "/home/hefangyuan/bag_files/bag3_arc/bag3.db3"
if not os.path.exists(bag_path):
    print('bag file not found')
else:
    print('no problem')
# 初始化坐标列表
x_coords, y_coords = [], []
# 打开 rosbag2 文件
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
reader.open(storage_options, rosbag2_py.ConverterOptions())

# 遍历消息
while reader.has_next():
    topic, data, timestamp = reader.read_next()
    if topic == "/odom_info":  # 替换为你的话题名
        msg = deserialize_message(data, Odometry)
        x_coords.append(msg.pose.pose.position.x)
        y_coords.append(msg.pose.pose.position.y)
    elif topic == "/path":  # 如果是路径数据
      msg = deserialize_message(data, Path)
      for pose in msg.poses:
        x_coords.append(pose.pose.position.x)
        y_coords.append(pose.pose.position.y)

# 绘制路径sq
plt.plot(x_coords, y_coords, 'b-', label="Robot Path")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("ROS 2 Navigation Path Visualization")
plt.legend()
plt.grid(True)
plt.axis('equal')  # 保持比例一致
plt.show()