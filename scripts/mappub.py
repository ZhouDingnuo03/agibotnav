#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import yaml
import os
import subprocess
import time
import signal
from collections import Counter

# ==================== 配置 ====================
MAP_DIR = "/home/orin-001/sda/agibotnav/grid_map_all/gird_map_to_pub"
YAML_NAME = "raw_grid_map.yaml"
PADDING = 30          # 有效区域外扩像素数
MIN_REGION_SIZE = 50  # 最小有效区域尺寸
# ==============================================

YAML_PATH = os.path.join(MAP_DIR, YAML_NAME)

def cleanup(signum, frame):
    subprocess.run(["pkill", "-f", "rviz2"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "static_transform_publisher"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rclpy.shutdown()
    except:
        pass
    print("\n 退出")
    exit(0)

signal.signal(signal.SIGINT, cleanup)


def auto_detect_valid_region(map_2d, padding=PADDING):
    height, width = map_2d.shape
    OBSTACLE_THRESHOLD = 50
    obstacle_mask = (map_2d >= OBSTACLE_THRESHOLD)

    if not np.any(obstacle_mask):
        print("[WARN] 未检测到障碍物，使用全图范围")
        return 0, width, 0, height

    valid_rows = np.any(obstacle_mask, axis=1)
    valid_cols = np.any(obstacle_mask, axis=0)

    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]

    y_min_raw = int(row_indices[0])
    y_max_raw = int(row_indices[-1]) + 1
    x_min_raw = int(col_indices[0])
    x_max_raw = int(col_indices[-1]) + 1

    region_h = y_max_raw - y_min_raw
    region_w = x_max_raw - x_min_raw

    if region_h < MIN_REGION_SIZE or region_w < MIN_REGION_SIZE:
        print(f"[WARN] 障碍区域过小，使用全图")
        return 0, width, 0, height

    x_min = max(0,      x_min_raw - padding)
    x_max = min(width,  x_max_raw + padding)
    y_min = max(0,      y_min_raw - padding)
    y_max = min(height, y_max_raw + padding)

    print(f"[INFO] 障碍范围：X=[{x_min_raw},{x_max_raw}] Y=[{y_min_raw},{y_max_raw}]")
    print(f"[INFO] 裁剪区域：X=[{x_min},{x_max}] Y=[{y_min},{y_max}]")
    print(f"[INFO] 新尺寸：{x_max-x_min} × {y_max-y_min}")
    return x_min, x_max, y_min, y_max

class MapPublisher(Node):
    def __init__(self):
        super().__init__("map_pub")
        self.pub = self.create_publisher(OccupancyGrid, "/map", 10)
        self.timer = self.create_timer(0.1, self.publish_map)
        self.map_msg = None
        self.load_map()

    def load_map(self):
        with open(YAML_PATH, 'r') as f:
            cfg = yaml.safe_load(f)

        # ┈┈┈ 完全使用你原来的代码 ┈┈┈
        data    = np.fromfile(os.path.join(MAP_DIR, cfg["data_file"]), dtype=np.int8)
        width   = cfg["width"]
        height  = cfg["height"]
        map_2d  = data.reshape(height, width)
        res     = 0.05

        # 值映射 10→20（你原来的逻辑）
        map_2d[map_2d == 10] = 20

        # ── 自适应裁剪（只保留有效区域）────────────────────
        x_min, x_max, y_min, y_max = auto_detect_valid_region(map_2d)
        cropped_map = map_2d[y_min:y_max, x_min:x_max]

        # 被裁掉的区域 → 彻底不发布
        new_w = cropped_map.shape[1]
        new_h = cropped_map.shape[0]

        # 原点修正（保证地图位置不变）
        origin_x = -50.0 + x_min * res
        origin_y = -50.0 + y_min * res

        # 统计
        flat_data = cropped_map.flatten()
        self.print_full_grid_statistics(flat_data)

        # 发布裁剪后的地图
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id     = "map"
        self.map_msg.info.width          = new_w
        self.map_msg.info.height         = new_h
        self.map_msg.info.resolution     = 0.05
        self.map_msg.info.origin.position.x      = origin_x
        self.map_msg.info.origin.position.y      = origin_y
        self.map_msg.info.origin.orientation.w   = 1.0
        self.map_msg.data = flat_data.tolist()

    def print_full_grid_statistics(self, data):
        print("\n" + "="*60)
        print(" 裁剪后栅格统计")
        print("="*60)
        total_cells = len(data)
        print(f"总栅格：{total_cells}")
        count_dict = Counter(data)
        for val in sorted(count_dict.keys()):
            cnt = count_dict[val]
            print(f"值={val:3d} | 数量={cnt:8d} | 占比={cnt/total_cells*100:5.2f}%")
        print("="*60 + "\n")

    def publish_map(self):
        if self.map_msg:
            self.map_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(self.map_msg)

def main():
    print("✅ 启动【自适应裁剪地图发布】")

    subprocess.run(["pkill", "-f", "rviz2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    subprocess.Popen([
        "ros2", "run", "tf2_ros", "static_transform_publisher",
        "0", "0", "0", "0", "0", "0", "map", "camera_init"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    rclpy.init()
    node = MapPublisher()

    print("✅ 启动 RViz2")
    subprocess.Popen(["rviz2"])

    print("\n地图发布中（仅发布有效区域）\n")
    rclpy.spin(node)

if __name__ == "__main__":
    main()
