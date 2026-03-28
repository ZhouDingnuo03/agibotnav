#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import os
import traceback
from datetime import datetime
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import yaml
import cv2
from scipy.ndimage import binary_dilation, binary_fill_holes

# 常量定义
OBSTACLE_CORE = 100
TABLE_MARK = 110

class MapSaver(Node):
    """AGIBOT D1 地图保存节点（覆盖式保存版）
    功能：
    1. 全局点云 PCD（覆盖保存）
    2. 原始栅格地图（覆盖保存）
    3. 后处理栅格地图（覆盖保存）
    无时间戳，每次直接覆盖旧文件
    """
    def __init__(self):
        super().__init__('agibot_map_saver')
        
        # ========== 核心配置 ==========
        self.global_map_topic = "/cumulative_cloud_registered"
        self.raw_map_topic = "/map"
        self.auto_save_interval = 5.0
        self.manual_save_topic = "/save_all_maps"
        
        # 保存路径
        self.base_save_path = "/home/orin-001/sda/agibotnav"
        self.pcd_save_path = os.path.join(self.base_save_path, "global_map")

        # 栅格地图保存到 grid_map_all/grid_map_N/ 分目录保存
        self.grid_map_all_base = os.path.join(self.base_save_path, "grid_map_all")
        # 自动找下一个可用编号：grid_map_1, grid_map_2...
        self.current_map_index = self._find_next_index()
        self.grid_save_path = os.path.join(self.grid_map_all_base, f"grid_map_{self.current_map_index}")
        # 创建当前目录
        os.makedirs(self.grid_save_path, exist_ok=True)

        # PCD文件名固定
        self.pcd_filename = "fastlio_global_map.pcd"
        self.raw_map_filename = "raw_grid_map"
        self.processed_map_filename = "processed_grid_map"
        
        # 配色配置
        self.img_unknown_color = (128, 128, 128)   # 未知-灰色
        self.img_obstacle_color = (0, 0, 255)      # 障碍-红色
        self.img_free_color = (255, 255, 255)      # 可通行-白色
        self.img_bottle_color = (0, 255, 0)        # 瓶子-绿色
        self.img_bottle_surround_color = (0, 200, 0)
        self.img_table_color = (255, 165, 0)        # 桌子-橙色
        self.img_wall_color = (138, 43, 226)        # 墙体-紫色
        
        # 栅格值配置
        self.COST_CAMERA_GROUND = 10
        self.COST_EMPTY_SPACE = 20
        self.COST_OBSTACLE = 100
        self.bottle_core_value = 120
        self.bottle_surround_value = 10
        
        # ========== 初始化变量 ==========
        self.global_points = np.array([])
        self.raw_grid_map = None
        
        # ========== 创建目录 ==========
        self._force_create_dirs()
        
        # ========== QoS配置 ==========
        pcd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        grid_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        manual_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # ========== 订阅话题 ==========
        try:
            self.global_map_sub = self.create_subscription(
                PointCloud2, self.global_map_topic, self.global_map_callback, pcd_qos
            )
            self.raw_map_sub = self.create_subscription(
                OccupancyGrid, self.raw_map_topic, self.raw_map_callback, grid_qos
            )
            self.manual_sub = self.create_subscription(
                PointCloud2, self.manual_save_topic, self.manual_save_callback, manual_qos
            )
        except Exception as e:
            self.get_logger().error(f"❌ 订阅话题失败：{str(e)}")
        
        # ========== 自动保存定时器 ==========
        try:
            self.auto_save_timer = self.create_timer(
                self.auto_save_interval, self.auto_save_callback
            )
        except Exception as e:
            self.get_logger().error(f"❌ 创建定时器失败：{str(e)}")
        
        # ========== 启动日志 ==========
        self.get_logger().info("="*60)
        self.get_logger().info(f"✅ AGIBOT地图保存节点启动成功")
        self.get_logger().info(f"📂 新地图将保存到：grid_map_all/grid_map_{self.current_map_index}")
        self.get_logger().info(f"💾 保存目录：{self.grid_save_path}")
        self.get_logger().info(f"💾 自动保存间隔：{self.auto_save_interval}秒")
        self.get_logger().info("="*60)

    # ==================== 工具函数 ====================
    def _find_next_index(self):
        """找到下一个可用的地图编号"""
        base = self.grid_map_all_base
        if not os.path.exists(base):
            return 1
        max_idx = 0
        for entry in os.listdir(base):
            if entry.startswith("grid_map_"):
                try:
                    idx = int(entry.split("_")[-1])
                    if idx > max_idx:
                        max_idx = idx
                except:
                    pass
        return max_idx + 1

    def _force_create_dirs(self):
        try:
            os.makedirs(self.pcd_save_path, exist_ok=True)
            os.makedirs(self.grid_save_path, exist_ok=True)
        except Exception as e:
            self.get_logger().fatal(f"❌ 创建目录失败：{str(e)}")
            raise

    # ==================== 回调函数 ====================
    def global_map_callback(self, msg):
        try:
            points_list = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([p[0], p[1], p[2]])
            if len(points_list) > 0:
                self.global_points = np.array(points_list)
        except Exception as e:
            self.get_logger().error(f"❌ 解析点云失败：{str(e)}")

    def raw_map_callback(self, msg):
        try:
            self.raw_grid_map = msg
        except Exception as e:
            self.get_logger().error(f"❌ 解析栅格失败：{str(e)}")

    # ==================== 后处理算法 ====================
    def detect_scored_walls(self, grid):
        obs = (grid == OBSTACLE_CORE).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        erode = cv2.erode(obs, kernel)
        edge = cv2.subtract(obs, erode)
        wall_out = np.zeros_like(edge)
        h, w = edge.shape
        segments = []
        for y in range(h):
            x = 0
            while x < w-5:
                if edge[y,x]:
                    cnt=1
                    while x+cnt<w and edge[y,x+cnt]:cnt+=1
                    if cnt>=6:
                        segments.append((cnt,0,x,y,x+cnt-1,y))
                    x += cnt
                else:
                    x += 1
        for x in range(w):
            y = 0
            while y < h-5:
                if edge[y,x]:
                    cnt=1
                    while y+cnt<h and edge[y+cnt,x]:cnt+=1
                    if cnt>=6:
                        segments.append((cnt,1,x,y,x,y+cnt-1))
                    y += cnt
                else:
                    y += 1
        candidates=[]
        for l,t,x1,y1,x2,y2 in segments:
            ang = np.rad2deg(np.arctan2(abs(y2-y1), abs(x2-x1)))
            if not ((ang<=15 or ang>=165) or (75<=ang<=105)):
                continue
            candidates.append((-l,ang,x1,y1,x2,y2))
        candidates.sort()
        used_h = []
        used_v = []
        for item in candidates:
            nl,ang,x1,y1,x2,y2 = item
            skip=False
            if ang<=15 or ang>=165:
                cy=y1
                for uy in used_h:
                    if abs(cy-uy)<=2:
                        skip=True
                if not skip:
                    used_h.append(cy)
            else:
                cx=x1
                for ux in used_v:
                    if abs(cx-ux)<=2:
                        skip=True
                if not skip:
                    used_v.append(cx)
            if not skip:
                cv2.line(wall_out,(x1,y1),(x2,y2),255,2)
        return wall_out

    def mark_tables_precise(self, grid, wall_mask):
        res = grid.copy()
        wall = wall_mask>0
        valid = (grid==OBSTACLE_CORE) & ~wall
        H,W = valid.shape
        rects = []
        for y0 in range(H):
            for x0 in range(W):
                if not valid[y0,x0]:
                    continue
                w=0
                while x0+w<W and valid[y0,x0+w]:
                    w+=1
                if w==0:
                    continue
                h=0
                while y0+h<H:
                    ok=True
                    for dx in range(w):
                        if not valid[y0+h, x0+dx]:
                            ok=False
                            break
                    if not ok:
                        break
                    h+=1
                if w*h>=12:
                    rects.append((-w*h,x0,y0,w,h))
        rects.sort()
        used = np.zeros_like(valid)
        for na,x0,y0,w,h in rects:
            conflict=False
            for dy in range(h):
                for dx in range(w):
                    if used[y0+dy,x0+dx]:
                        conflict=True
                        break
                if conflict:
                    break
            if not conflict:
                for dy in range(h):
                    for dx in range(w):
                        res[y0+dy,x0+dx] = TABLE_MARK
                        used[y0+dy,x0+dx] = True
        return res

    def dilate_obstacle(self, grid):
        m = grid == OBSTACLE_CORE
        g = binary_dilation(m, np.ones((3,3), bool))
        out = np.zeros_like(grid)
        out[g] = OBSTACLE_CORE
        return out

    def fill_holes(self, grid):
        m = grid == OBSTACLE_CORE
        g = binary_fill_holes(m)
        out = np.zeros_like(grid)
        out[g] = OBSTACLE_CORE
        return out

    # ==================== 完整后处理流程 ====================
    def process_grid_map(self, raw_grid_msg):
        try:
            grid = np.array(raw_grid_msg.data, dtype=np.int8).reshape(
                raw_grid_msg.info.height, raw_grid_msg.info.width
            )
            wall_mask = self.detect_scored_walls(grid)
            grid = self.mark_tables_precise(grid, wall_mask)
            grid = self.dilate_obstacle(grid)
            grid = self.fill_holes(grid)
            
            processed_msg = OccupancyGrid()
            processed_msg.header = raw_grid_msg.header
            processed_msg.info = raw_grid_msg.info
            processed_msg.data = grid.flatten().tolist()
            return processed_msg
        except Exception as e:
            self.get_logger().error(f"❌ 后处理失败：{str(e)}")
            return None

    # ==================== 栅格转图片 ====================
    def _grid_to_image(self, grid_map, is_processed=False):
        if grid_map is None:
            return None
        
        try:
            grid_data = np.array(grid_map.data, dtype=np.int8).reshape(
                grid_map.info.height, grid_map.info.width
            )
            img = np.full((grid_map.info.height, grid_map.info.width, 3),
                         self.img_unknown_color, dtype=np.uint8)
            
            free_mask = (grid_data == self.COST_CAMERA_GROUND) | (grid_data == self.COST_EMPTY_SPACE)
            img[free_mask] = self.img_free_color
            img[grid_data == self.COST_OBSTACLE] = self.img_obstacle_color
            img[grid_data == self.bottle_core_value] = self.img_bottle_color
            img[grid_data == self.bottle_surround_value] = self.img_bottle_surround_color
            
            if is_processed:
                img[grid_data == TABLE_MARK] = self.img_table_color
                wall_mask = self.detect_scored_walls(grid_data)
                img[wall_mask > 0] = self.img_wall_color
            
            img[grid_data == -1] = self.img_unknown_color
            return img
        except Exception as e:
            self.get_logger().error(f"❌ 栅格转图片失败：{str(e)}")
            return None

    # ==================== 覆盖式保存函数 ====================
    def _save_pcd(self):
        try:
            if len(self.global_points) == 0:
                self.get_logger().warn("⚠️ 无点云数据")
                return False
            
            save_path = os.path.join(self.pcd_save_path, self.pcd_filename)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.global_points)
            o3d.io.write_point_cloud(save_path, pcd)
            self.get_logger().info(f"✅ 全局点云已保存：{save_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ 保存PCD失败：{str(e)}")
            return False

    def _save_grid_image(self, grid_map, base_filename, is_processed=False):
        try:
            img = self._grid_to_image(grid_map, is_processed)
            if img is None:
                return False
            img_path = os.path.join(self.grid_save_path, f"{base_filename}.png")
            cv2.imwrite(img_path, img)
            return True
        except Exception as e:
            self.get_logger().error(f"❌ 保存图片失败：{str(e)}")
            return False

    def _save_grid_map(self, grid_map, base_filename, is_processed=False):
        try:
            if grid_map is None:
                return False
            
            yaml_path = os.path.join(self.grid_save_path, f"{base_filename}.yaml")
            bin_path = os.path.join(self.grid_save_path, f"{base_filename}.bin")
            
            np.array(grid_map.data, dtype=np.int8).tofile(bin_path)
            
            yaml_data = {
                "resolution": grid_map.info.resolution,
                "origin_x": grid_map.info.origin.position.x,
                "origin_y": grid_map.info.origin.position.y,
                "width": grid_map.info.width,
                "height": grid_map.info.height,
                "data_file": f"{base_filename}.bin",
                "frame_id": grid_map.header.frame_id,
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_data, f, indent=2)
            
            self._save_grid_image(grid_map, base_filename, is_processed)
            return True
        except Exception as e:
            self.get_logger().error(f"❌ 保存栅格失败：{str(e)}")
            return False

    # ==================== 自动/手动保存（全覆盖） ====================
    def auto_save_callback(self):
        self.get_logger().info("\n🔄 自动覆盖保存中...")
        self._save_pcd()
        self._save_grid_map(self.raw_grid_map, self.raw_map_filename, False)
        
        processed_map = self.process_grid_map(self.raw_grid_map)
        self._save_grid_map(processed_map, self.processed_map_filename, True)
        self.get_logger().info("✅ 全部覆盖完成\n")

    def manual_save_callback(self, msg):
        self.get_logger().info("\n📢 手动覆盖保存中...")
        self.auto_save_callback()

def main(args=None):
    try:
        rclpy.init(args=args)
        node = MapSaver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 节点已停止")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

