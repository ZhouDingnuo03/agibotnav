#!/usr/bin/env python3
"""
04_inference.py  —  订阅 /map，用训练好的 UNet 推理并发布语义地图（多标签4类）

发布:
  /semantic_map   (nav_msgs/OccupancyGrid) - 单通道最大值
  /semantic_map_img (sensor_msgs/Image)  BGR8 彩色可视化（支持多标签混合）
  /semantic_labels (sensor_msgs/Image)  4通道二值标签 [房间,走廊,墙壁,其他]

用法:
  python3 04_inference.py --model checkpoints/best.pth
  python3 04_inference.py --model checkpoints/best.pth --base_ch 16
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
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
#  颜色 & 语义映射（4类多标签）
# ══════════════════════════════════════════════════════════
#  模型输出 4通道: 0=房间  1=走廊  2=墙壁  3=其他
#  背景通过所有通道都<0.5表示

NUM_CLASSES = 4
LABEL_NAMES = ["房间", "走廊", "墙壁", "其他"]

# BGR 彩色（用于 Image 话题）
LABEL_COLORS = np.array([
    [  0, 200,   0],   # 0 房间  绿
    [  0, 220, 220],   # 1 走廊  青
    [ 60,  60, 220],   # 2 墙壁  红
    [  0, 140, 255],   # 3 其他  橙
    [100, 100, 100],   # 背景  灰
], dtype=np.uint8)

# 用于单通道 OccupancyGrid 发布（取最大值）
SEG2LABEL = {0: 10, 1: 50, 2: 90, 3: 30}

WALL_THR = 50          # 占用格栅墙壁阈值

# ══════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════

def norm_occ(occ: np.ndarray) -> np.ndarray:
    """int16 占用格 → float32 [0,1]；-1 → 0.5"""
    out = np.full(occ.shape, 0.5, dtype=np.float32)
    valid = occ >= 0
    out[valid] = occ[valid].astype(np.float32) / 100.0
    return out


def pad_to_multiple(img: np.ndarray, multiple: int = 16
                    ) -> tuple[np.ndarray, tuple[int,int]]:
    """将 HW 图像 pad 到 multiple 的倍数，返回 (padded, (ph, pw))"""
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


def colorize_multilabel(pred_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    多标签预测结果彩色化
    pred_probs: (4, H, W) float32, sigmoid 输出 [0,1]
    返回: (H, W, 3) uint8 BGR
    """
    c, h, w = pred_probs.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    has_label = np.zeros((h, w), dtype=bool)
    for cls in range(NUM_CLASSES):
        mask = pred_probs[cls] > threshold
        has_label |= mask
        if np.any(mask):
            out[mask] = np.clip(out[mask].astype(np.float32) * 0.5 + LABEL_COLORS[cls] * 0.5, 0, 255).astype(np.uint8)

    # 背景（所有标签都<0.5）
    bg_mask = ~has_label
    out[bg_mask] = LABEL_COLORS[4]

    return out


# ══════════════════════════════════════════════════════════
#  推理节点
# ══════════════════════════════════════════════════════════

class SemanticMapNode(Node):
    def __init__(self, model_path: str, base_ch: int, device_str: str,
                 throttle_hz: float, threshold: float = 0.5):
        super().__init__("semantic_map_node")
        self.device = torch.device(device_str)
        self.threshold = threshold

        # ── 加载模型 ──────────────────────────────────────
        self.get_logger().info(f"加载模型: {model_path}")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        # 直接加载 state_dict（03_train.py 只保存了 state_dict）
        bc = base_ch
        self.model = UNet(in_ch=1, num_classes=4, base=bc).to(self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.get_logger().info(f"✅ 模型加载完成  base={bc}  阈值={threshold}")

        # ── 发布 / 订阅 ───────────────────────────────────
        self._pub_grid = self.create_publisher(
            OccupancyGrid, "/semantic_map", 10)
        self._pub_img  = self.create_publisher(
            Image,         "/semantic_map_img", 10)
        self._pub_labels = self.create_publisher(
            Image,         "/semantic_labels", 10)
        self._sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_cb, 10)

        # 限流
        self._min_interval = 1.0 / max(throttle_hz, 0.1)
        self._last_t       = 0.0

        self.get_logger().info(
            f"订阅 /map  →  发布 /semantic_map + /semantic_map_img + /semantic_labels\n"
            f"设备: {device_str}  限频: {throttle_hz:.1f} Hz"
        )

    # ── 主回调 ────────────────────────────────────────────
    def _map_cb(self, msg: OccupancyGrid):
        now = time.monotonic()
        if now - self._last_t < self._min_interval:
            return
        self._last_t = now

        t0 = time.perf_counter()

        # ── 解析消息 ──────────────────────────────────────
        h, w = msg.info.height, msg.info.width
        occ  = np.array(msg.data, dtype=np.int16).reshape(h, w)

        # Grid 原点在左下；模型训练时做了 flipud；推理同样 flipud→推理→flipud 回去
        occ_flip = np.flipud(occ)

        # ── 预处理 ────────────────────────────────────────
        img = norm_occ(occ_flip)                        # float32 HW
        img_pad, (ph, pw) = pad_to_multiple(img, 16)   # pad

        tensor = (torch.from_numpy(img_pad)
                  .unsqueeze(0).unsqueeze(0)            # 1×1×H'×W'
                  .to(self.device))

        # ── 推理 ──────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(tensor)                 # 1×4×H'×W'
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # 4×H'×W'

        # 裁掉 padding
        pred_probs = pred_probs[:, :h, :w]             # 4×H×W

        # 单通道预测（取最大值）用于 OccupancyGrid
        pred_max = pred_probs.argmax(axis=0)            # H×W
        # 低置信度设为背景
        max_prob = pred_probs.max(axis=0)
        pred_max[max_prob < self.threshold] = 255

        # -1（unknown） 区域设为 unknown_class; 依据原始 occ
        unknown_mask = (occ_flip < 0)
        pred_max[unknown_mask] = 255

        # flipud 恢复 Grid 坐标（左下原点）
        pred_grid = np.flipud(pred_max).astype(np.int32)

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"推理完成  {w}×{h}  {elapsed:.1f} ms",
            throttle_duration_sec=2.0
        )

        # ── 发布 OccupancyGrid ────────────────────────────
        self._publish_grid(pred_grid, msg)

        # ── 发布彩色图像（多标签混合） ─────────────────────
        self._publish_image(pred_probs, msg.header)

        # ── 发布4通道二值标签 ─────────────────────────────
        self._publish_labels(pred_probs, unknown_mask, msg.header)

    def _publish_grid(self, pred_grid: np.ndarray, src: OccupancyGrid):
        """pred_grid: HW, Grid坐标（左下原点），值 0/1/2/3/255"""
        h, w = pred_grid.shape
        flat = np.full(h * w, -1, dtype=np.int8)

        for seg, val in SEG2LABEL.items():
            flat[pred_grid.ravel() == seg] = val
        # 255(未知) 保持 -1

        msg = OccupancyGrid()
        msg.header.stamp    = now_stamp(self)
        msg.header.frame_id = src.header.frame_id
        msg.info            = src.info
        msg.data            = flat.tolist()
        self._pub_grid.publish(msg)

    def _publish_image(self, pred_probs: np.ndarray, src_header: Header):
        """pred_probs: (4, H, W), 图像坐标（左上原点）"""
        bgr = colorize_multilabel(pred_probs, self.threshold)
        h, w = bgr.shape[:2]

        img_msg = Image()
        img_msg.header.stamp    = now_stamp(self)
        img_msg.header.frame_id = src_header.frame_id
        img_msg.height    = h
        img_msg.width     = w
        img_msg.encoding  = "bgr8"
        img_msg.is_bigendian = False
        img_msg.step      = w * 3
        img_msg.data      = bgr.tobytes()
        self._pub_img.publish(img_msg)

    def _publish_labels(self, pred_probs: np.ndarray, unknown_mask: np.ndarray, src_header: Header):
        """发布4通道二值标签，每个通道0-255表示置信度"""
        c, h, w = pred_probs.shape
        # 转为 uint8 [0, 255]
        labels_uint8 = (pred_probs * 255).astype(np.uint8)
        # 未知区域设为0
        labels_uint8[:, unknown_mask] = 0
        # 转置为 (H, W, C) 格式
        labels_hwc = np.transpose(labels_uint8, (1, 2, 0))

        img_msg = Image()
        img_msg.header.stamp    = now_stamp(self)
        img_msg.header.frame_id = src_header.frame_id
        img_msg.height    = h
        img_msg.width     = w
        img_msg.encoding  = "mono8" if c == 1 else "8UC4"
        img_msg.is_bigendian = False
        img_msg.step      = w * c
        img_msg.data      = labels_hwc.tobytes()
        self._pub_labels.publish(img_msg)


# ══════════════════════════════════════════════════════════
#  离线推理（无 ROS2，直接处理 .npy 文件）
# ══════════════════════════════════════════════════════════

def infer_offline(args):
    """离线模式：读取 map_data/ 下所有地图并输出 PNG"""
    import glob, os
    import cv2

    device = torch.device(args.device)
    # 直接加载 state_dict
    model = UNet(in_ch=1, num_classes=4, base=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"模型加载完成  base={args.base_ch}")

    all_npy = sorted(glob.glob(os.path.join(args.data_dir, "map_*.npy")))
    all_npy = [f for f in all_npy if "_label" not in f]
    if not all_npy:
        print(f"找不到 {args.data_dir}/map_*.npy")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for path in all_npy:
        occ      = np.flipud(np.load(path).astype(np.int16))
        h, w     = occ.shape
        img, (ph, pw) = pad_to_multiple(norm_occ(occ), 16)
        tensor   = (torch.from_numpy(img)
                    .unsqueeze(0).unsqueeze(0).to(device))
        with torch.no_grad():
            logits = model(tensor)
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()[:, :h, :w]

        # 彩色图（多标签混合）
        bgr = colorize_multilabel(pred_probs, args.threshold)

        # 拼接原图 + 预测
        raw_vis = np.zeros((h, w, 3), dtype=np.uint8)
        raw_vis[:]          = 100
        raw_vis[occ == 0]   = 220
        raw_vis[occ >= WALL_THR] = 35
        m = (occ > 0) & (occ < WALL_THR)
        v = np.clip(220 - occ[m].astype(np.float32)*1.8, 60, 220).astype(np.uint8)
        raw_vis[m] = np.stack([v, v, v], axis=-1)

        combo = np.hstack([raw_vis, bgr])
        base  = os.path.splitext(os.path.basename(path))[0]
        out   = os.path.join(args.output_dir, f"{base}_semantic.png")
        cv2.imwrite(out, combo)
        print(f"  ✅ {out}")

    print("离线推理完成")


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("04_inference.py")
    ap.add_argument("--model",       default="checkpoints/best.pth",
                    help="模型权重路径，e.g. checkpoints/best.pth")
    ap.add_argument("--base_ch",     type=int, default=32,
                    help="UNet 基础通道")
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available()
                                               else "cpu")
    ap.add_argument("--throttle_hz", type=float, default=2.0,
                    help="ROS2 模式下最高推理频率")
    ap.add_argument("--threshold",   type=float, default=0.5,
                    help="多标签分类阈值")
    # 离线模式
    ap.add_argument("--offline",     action="store_true",
                    help="离线模式：不启动 ROS2，直接处理 --data_dir 下的地图")
    ap.add_argument("--data_dir",    default="map_data")
    ap.add_argument("--output_dir",  default="inference_results")

    args, _ = ap.parse_known_args()

    if args.offline:
        infer_offline(args)
        return

    # ── ROS2 模式 ─────────────────────────────────────────
    rclpy.init()
    node = SemanticMapNode(
        model_path  = args.model,
        base_ch     = args.base_ch,
        device_str  = args.device,
        throttle_hz = args.throttle_hz,
        threshold   = args.threshold,
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
