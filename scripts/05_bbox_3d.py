#!/usr/bin/env python3
"""
05_bbox_3d.py  —  语义分割后对每个识别区域计算Bounding Box，发布3D框用于RViz显示

功能：
  - 订阅 /map (OccupancyGrid)
  - UNet推理得到4类语义分割结果（房间、走廊、墙壁、其他）
  - 对每一类找出连通区域，计算每个区域的bounding box
  - 发布 /semantic_bboxes 作为 MarkerArray，每个区域用不同颜色框出
  - 发布 /semantic_centroids 点云显示每个区域中心点

用法:
  python3 05_bbox_3d.py --model checkpoints/best.pth
"""

import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header
from builtin_interfaces.msg import Time as RosTime

# ══════════════════════════════════════════════════════════
#  复制模型定义（与 03_train.py 保持一致）
# ══════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c_in, c_out))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in//2, 2, 2)
        self.conv = DoubleConv(c_in//2 + c_skip, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        dh, dw = skip.shape[2]-x.shape[2], skip.shape[3]-x.shape[3]
        x = F.pad(x, (0,dw,0,dh))
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        b = base
        self.e1 = DoubleConv(in_ch, b)
        self.e2 = Down(b,   b*2)
        self.e3 = Down(b*2, b*4)
        self.e4 = Down(b*4, b*8)
        self.bot = DoubleConv(b*8, b*8)
        self.d3 = Up(b*8, b*4, b*4)
        self.d2 = Up(b*4, b*2, b*2)
        self.d1 = Up(b*2, b,   b)
        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        x = self.bot(e4)
        x = self.d3(x, e3)
        x = self.d2(x, e2)
        x = self.d1(x, e1)
        return self.head(x)

# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════
NUM_CLASSES = 4
LABEL_NAMES = ["房间", "走廊", "墙壁", "其他"]

# BGR → RGB for ROS markers
LABEL_COLORS_RGB = [
    [  0, 200,   0],   # 0 房间  绿
    [  0, 220, 220],   # 1 走廊  青
    [ 60,  60, 220],   # 2 墙壁  蓝
    [  0, 140, 255],   # 3 其他  橙
]

MIN_REGION_SIZE = 50  # 最小区域像素数，太小忽略

# ══════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════

def norm_occ(occ: np.ndarray) -> np.ndarray:
    out = np.full(occ.shape, 0.5, dtype=np.float32)
    valid = occ >= 0
    out[valid] = occ[valid].astype(np.float32) / 100.0
    return out

def pad_to_multiple(img: np.ndarray, multiple: int = 16
                    ) -> tuple[np.ndarray, tuple[int,int]]:
    h, w = img.shape
    ph   = (multiple - h % multiple) % multiple
    pw   = (multiple - w % multiple) % multiple
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    return img, (ph, pw)

def now_stamp(node: Node) -> RosTime:
    t = node.get_clock().now()
    msg = RosTime()
    msg.sec, msg.nanosec = t.seconds_nanoseconds()
    return msg

def find_connected_regions(mask: np.ndarray, min_size: int = MIN_REGION_SIZE):
    """找出二值mask中的连通区域，返回每个区域的bounding box (y1, x1, y2, x2)"""
    from scipy.ndimage import label, find_objects
    labeled, num = label(mask)
    regions = []
    slices = find_objects(labeled)
    for i, sl in enumerate(slices):
        y1, y2 = sl[0].start, sl[0].stop
        x1, x2 = sl[1].start, sl[1].stop
        area = (y2 - y1) * (x2 - x1)
        if area >= min_size:
            regions.append((y1, x1, y2, x2, area))
    return regions

# ══════════════════════════════════════════════════════════
#  Bounding Box 发布节点
# ══════════════════════════════════════════════════════════

class SemanticBBoxNode(Node):
    def __init__(self, model_path: str, base_ch: int, device_str: str,
                 throttle_hz: float, threshold: float, height_3d: float = 1.0):
        super().__init__("semantic_bbox_node")
        self.device = torch.device(device_str)
        self.threshold = threshold
        self.height_3d = height_3d  # 3D框在Z方向的高度

        # ── 加载模型 ──────────────────────────────────────
        self.get_logger().info(f"加载模型: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        bc = base_ch
        self.model = UNet(in_ch=1, num_classes=4, base=bc).to(self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.get_logger().info(f"✅ 模型加载完成  base={bc}  阈值={threshold}")

        # ── 发布 / 订阅 ───────────────────────────────────
        self._sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_cb, 10)
        self._pub_bboxes = self.create_publisher(
            MarkerArray, "/semantic_bboxes", 10)
        self._pub_centroids = self.create_publisher(
            Marker, "/semantic_centroids", 10)

        # 限流
        self._min_interval = 1.0 / max(throttle_hz, 0.1)
        self._last_t       = 0.0

        self.get_logger().info(
            f"订阅 /map  →  发布 /semantic_bboxes (MarkerArray) + /semantic_centroids (point cloud)\n"
            f"设备: {device_str}  限频: {throttle_hz:.1f} Hz  最小区域: {MIN_REGION_SIZE} 像素"
        )

    def _grid_to_3d(self, grid_y: int, grid_x: int, info):
        """栅格坐标 → 世界XY坐标"""
        x = info.origin.position.x + (grid_x + 0.5) * info.resolution
        y = info.origin.position.y + (grid_y + 0.5) * info.resolution
        return x, y

    def _create_box_marker(self, marker_id: int, cls: int,
                          y1: int, x1: int, y2: int, x2: int, frame_id: str, msg_info,
                          timestamp):
        """创建一个cube marker表示bounding box"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = f"semantic_{LABEL_NAMES[cls]}"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 中心点坐标
        cx_pix = (x1 + x2) / 2.0
        cy_pix = (y1 + y2) / 2.0
        cx, cy = self._grid_to_3d(int(cy_pix), int(cx_pix), info)
        cz = self.height_3d / 2.0

        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = cz
        marker.pose.orientation.w = 1.0

        # 尺寸
        dx = (x2 - x1) * info.resolution
        dy = (y2 - y1) * info.resolution
        dz = self.height_3d

        marker.scale.x = dx
        marker.scale.y = dy
        marker.scale.z = dz

        # 颜色
        color = LABEL_COLORS_RGB[cls]
        marker.color.r = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.b = color[2] / 255.0
        marker.color.a = 0.3  # 半透明

        return marker

    def _map_cb(self, msg: OccupancyGrid):
        now = time.monotonic()
        if now - self._last_t < self._min_interval:
            return
        self._last_t = now

        t0 = time.perf_counter()

        # ── 解析消息 ──────────────────────────────────────
        h, w = msg.info.height, msg.info.width
        occ  = np.array(msg.data, dtype=np.int16).reshape(h, w)

        # Grid 原点在左下；flipud 对齐训练坐标
        occ_flip = np.flipud(occ)

        # ── 预处理 ────────────────────────────────────────
        img = norm_occ(occ_flip)
        img_pad, (ph, pw) = pad_to_multiple(img, 16)
        tensor = (torch.from_numpy(img_pad)
                  .unsqueeze(0).unsqueeze(0)
                  .to(self.device))

        # ── 推理 ──────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(tensor)
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # 裁掉 padding
        pred_probs = pred_probs[:, :h, :w]  # 4×H×W

        # ── 对每个类别找连通区域计算 bounding box ────────
        marker_array = MarkerArray()
        centroid_points = Marker()
        centroid_points.header.frame_id = msg.header.frame_id
        centroid_points.header.stamp = now_stamp(self)
        centroid_points.ns = "centroids"
        centroid_points.id = 0
        centroid_points.type = Marker.POINTS
        centroid_points.action = Marker.ADD
        centroid_points.scale.x = 0.2
        centroid_points.scale.y = 0.2

        marker_id = 0
        total_regions = 0

        for cls in range(NUM_CLASSES):
            mask = pred_probs[cls] > self.threshold
            if not np.any(mask):
                continue

            regions = find_connected_regions(mask)
            for (y1, x1, y2, x2, area) in regions:
                # 创建 bounding box marker
                marker = self._create_box_marker(
                    marker_id, cls, y1, x1, y2, x2, msg.header.frame_id, msg.info, now_stamp(self))
                marker_array.markers.append(marker)
                marker_id += 1

                # 添加中心点
                cx_pix = (x1 + x2) / 2.0
                cy_pix = (y1 + y2) / 2.0
                cx, cy = self._grid_to_3d(int(cy_pix), int(cx_pix), msg.info)
                p = Point()
                p.x = cx
                p.y = cy
                p.z = self.height_3d + 0.1
                centroid_points.points.append(p)
                total_regions += 1

        # ── 发布 ──────────────────────────────────────────
        # 清除旧markers：先delete all，再add新的
        if len(marker_array.markers) == 0:
            # 添加一个delete all marker
            del_marker = Marker()
            del_marker.header.frame_id = msg.header.frame_id
            del_marker.header.stamp = now_stamp(self)
            del_marker.action = Marker.DELETEALL
            marker_array.markers.append(del_marker)

        self._pub_bboxes.publish(marker_array)
        self._pub_centroids.publish(centroid_points)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"推理完成  {w}×{h}  检测到 {total_regions} 个区域  {elapsed:.1f} ms",
            throttle_duration_sec=2.0
        )

# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("05_bbox_3d.py")
    ap.add_argument("--model",       default="checkpoints/best.pth",
                    help="模型权重路径，e.g. checkpoints/best.pth")
    ap.add_argument("--base_ch",     type=int, default=32,
                    help="UNet 基础通道")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available()
                                               else "cpu")
    ap.add_argument("--throttle_hz", type=float, default=2.0,
                    help="最高推理频率")
    ap.add_argument("--threshold",   type=float, default=0.5,
                    help="多标签分类阈值")
    ap.add_argument("--height_3d",   type=float, default=2.0,
                    help="3D bounding box高度 (米)")
    ap.add_argument("--min_region",  type=int, default=50,
                    help="最小区域像素数，小于此忽略")

    args, _ = ap.parse_known_args()

    global MIN_REGION_SIZE
    MIN_REGION_SIZE = args.min_region

    # ── ROS2 模式 ─────────────────────────────────────────
    rclpy.init()
    node = SemanticBBoxNode(
        model_path  = args.model,
        base_ch     = args.base_ch,
        device_str  = args.device,
        throttle_hz = args.throttle_hz,
        threshold   = args.threshold,
        height_3d   = args.height_3d,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
