import rosbag2_py
import cv2
import os
import numpy as np
from sensor_msgs.msg import CompressedImage
import h5py
from glob import glob
from natsort import natsorted
from typing import Dict, List
from PIL import Image
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# -------------------------------
# 设置参数和路径
# -------------------------------
bag_path = '/home/hefangyuan/bag_files/trace/1/1.db3'  
bag_name = os.path.splitext(os.path.basename(bag_path))[0]
topics = ['/img/CAM_A/compressed']
target_freq = 2.0  # Hz
draw_text = False  # <<<<<<<<<<<<<<<<< Control whether to add labels!!
output_root = os.path.join('sampled_images', bag_name)
os.makedirs(output_root, exist_ok=True)

start_frame = 340   # Start frame (inclusive)
end_frame = 410     # End frame (inclusive)

frame_id_to_label = {
    340: "proceed along the road",
    357: "go around the obstacle",
    363: "proceed along the road",
    380: "go around the person in front",
    387: "proceed along the road",
    393: "pass through the crowd",
}

overall_command = "navigation_task"

# -------------------------------
# 功能函数
# -------------------------------

def get_label_from_id(frame_id: int, mapping: Dict[int, str]) -> str:
    keys = sorted(mapping.keys())
    for i, key in enumerate(keys):
        if frame_id < key:
            return mapping[keys[i-1]] if i > 0 else mapping[key]
    return mapping[keys[-1]]

def get_text_color(image, region=(0, 0, 100, 40)):
    x, y, w, h = region
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return (0, 0, 0) if avg_brightness > 128 else (255, 255, 255)

def draw_label_on_image(image, label):
    font_scale = 1.2
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    padding_x = 20
    padding_y = 15
    bg_x2 = text_w + padding_x * 2
    bg_y2 = text_h + padding_y * 2

    overlay = image.copy()
    text_color = get_text_color(image, region=(0, 0, bg_x2, bg_y2))
    bg_color = (0, 0, 0) if text_color == (255, 255, 255) else (255, 255, 255)

    cv2.rectangle(overlay, (0, 0), (bg_x2, bg_y2), bg_color, -1)
    image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    cv2.putText(image, label, (padding_x, padding_y + text_h), font, font_scale, text_color, thickness)
    return image

# -------------------------------
# 主逻辑
# -------------------------------

def extract_and_process():
    all_images = []
    all_filenames = []
    all_labels = []

    # Create reader instance and open the bag file
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic metadata
    topic_types = reader.get_all_topics_and_types()
    
    # Create a map for quicker lookup
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    for topic in topics:
        topic_safe = topic.replace('/', '_')
        output_dir = os.path.join(output_root, topic_safe)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[{topic}] Starting processing...")

        last_saved_time = None
        count = 0

        # Set the reader's filter for the current topic
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

        while reader.has_next():
            (topic_name, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic_name])
            msg = deserialize_message(data, msg_type)

            curr_time = t / 1e9  # Convert nanoseconds to seconds
            if (last_saved_time is None) or (curr_time - last_saved_time >= (1.0 / target_freq)):
                if hasattr(msg, 'data'):  # Check if it's a CompressedImage
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if image is not None:
                        label = get_label_from_id(count, frame_id_to_label)
                        filename = f"{count:04d}.jpg"
                        filepath = os.path.join(output_dir, filename)

                        if start_frame <= count <= end_frame:
                            # Save image or add to list
                            if draw_text:
                                image = draw_label_on_image(image, label)
                                cv2.imwrite(filepath, image)  # Save labeled image
                            else:
                                all_images.append(image)
                                all_filenames.append(filename)
                                all_labels.append(label)

                            if draw_text:
                                print(f"✅ Saved labeled image {filename} with label {label}")
                            else:
                                print(f"✅ Captured original image {filename} with label {label}")
                        else:
                            # Skip frames outside the range
                            pass
                        
                        last_saved_time = curr_time
                        count += 1

        print(f"🎉 Completed extraction of {count} frames")

        if draw_text:
            # Generate video and GIF
            image_files = natsorted(glob(os.path.join(output_dir, '*.jpg')))
            image_files = [f for f in image_files if start_frame <= int(os.path.basename(f).split('.')[0]) <= end_frame]

            if image_files:
                first_frame = cv2.imread(image_files[0])
                height, width, _ = first_frame.shape

                video_path = os.path.join(output_dir, f"{bag_name}.mp4")
                gif_path = os.path.join(output_dir, f"{bag_name}.gif")

                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), target_freq, (width, height))
                gif_frames = []

                for img_path in image_files:
                    frame = cv2.imread(img_path)
                    video_writer.write(frame)
                    gif_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                video_writer.release()

                gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=int(1000/target_freq), loop=0)

                print(f"✅ Video and GIF generated: {video_path}, {gif_path}")
            else:
                print("⚠️ No images available for video generation")

        else:
            # Save to h5 file
            h5_path = os.path.join(output_root, f"{bag_name}_{topic_safe}.h5")
            with h5py.File(h5_path, 'w') as h5f:
                images_array = np.stack(all_images)  # (N, H, W, 3)
                filenames_array = np.array(all_filenames, dtype='S')
                labels_array = np.array(all_labels, dtype='S')

                h5f.create_dataset('images', data=images_array, compression='gzip')
                h5f.create_dataset('filenames', data=filenames_array)
                h5f.create_dataset('labels', data=labels_array)
                h5f.attrs['command'] = overall_command.encode('utf-8')
            print(f"✅ Images and labels saved to HDF5: {h5_path}")


if __name__ == "__main__":
    extract_and_process()