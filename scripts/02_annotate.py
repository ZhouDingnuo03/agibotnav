#!/usr/bin/env python3
"""
02_annotate.py — 任意四边形语义地图标注工具（支持多标签共存）

标注方式：
  每个类别独立渲染到自己的通道，不同类别的四边形可以重叠
  同一像素可以同时属于多个类别（房间里有走廊、墙壁等）

标签:
  0: 未标注（无四边形覆盖的区域）
  1: 房间  2: 走廊  3: 墙壁  4: 其他

鼠标:
  左键单击   依次点击四个顶点绘制四边形（第四点自动完成）
  右键单击   完成当前四边形（点数>=3）或删除被点击的四边形
  滚轮       缩放（以鼠标为中心）
  中键拖动   平移

键盘:
  1  房间(绿)   2  走廊(青)   3  墙壁(红)   4  其他(橙)   0  清除选中标签
  F  插入"全图墙壁"底层四边形（快速填底）
  Z  撤销（删除当前点或最后一个四边形）
  C  清空所有四边形
  S  保存（_quads.json + _label.npy）
  N / P  下一张 / 上一张地图（自动保存）
  Q / Esc  退出（自动保存）
"""
import numpy as np
import cv2
import os, glob, json, argparse
from typing import List, Tuple, Optional

# ── Pillow 中文字体支持 ────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False
    print("[警告] 未安装 Pillow，中文将显示为问号。请运行: pip install pillow")

WIN_W, WIN_H = 1280, 800
WALL_THR     = 50
RECT_ALPHA   = 0.40     # 四边形填充透明度

# ── 标签定义 ──────────────────────────────────────────────
LABELS    = {0: "清除", 1: "房间", 2: "走廊", 3: "墙壁", 4: "其他"}
LABELS_EN = {0: "Clear", 1: "Room", 2: "Corridor", 3: "Wall", 4: "Other"}
COLORS    = {           # BGR
    0: (100, 100, 100),  # 灰  清除
    1: (  0, 200,   0),  # 绿  房间
    2: (  0, 220, 220),  # 青  走廊
    3: ( 60,  60, 220),  # 红  墙壁
    4: (  0, 140, 255),  # 橙  其他
}


# ── 字体缓存 ──────────────────────────────────────────────
_FONT_CACHE: dict = {}

def _get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = ImageFont.load_default()
    for path in [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            pass
    _FONT_CACHE[size] = font
    return font


def draw_texts(canvas: np.ndarray, items: List[Tuple]) -> np.ndarray:
    if not items:
        return canvas
    if not _PIL:
        for text, (x, y), sz, bgr in items:
            cv2.putText(canvas, text, (x, y + sz),
                        cv2.FONT_HERSHEY_SIMPLEX, sz / 30.0, bgr, 1, cv2.LINE_AA)
        return canvas
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, (x, y), sz, bgr in items:
        draw.text((x, y), text, font=_get_font(sz),
                  fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── 地图可视化 ─────────────────────────────────────────────
def occ_to_bgr(occ: np.ndarray) -> np.ndarray:
    bgr = np.full((*occ.shape, 3), 100, dtype=np.uint8)
    bgr[occ == 0] = 220
    bgr[occ >= WALL_THR] = 35
    m = (occ > 0) & (occ < WALL_THR)
    v = np.clip(220 - occ[m].astype(np.float32) * 1.8, 60, 220).astype(np.uint8)
    bgr[m] = np.stack([v, v, v], axis=-1)
    return bgr


# ══════════════════════════════════════════════════════════
class QuadAnnotator:
    """
    多标签四边形标注器
    每个类别独立存储四边形列表，保存时渲染到不同通道
    """

    def __init__(self, data_dir: str):
        all_npy = sorted(glob.glob(os.path.join(data_dir, "map_*.npy")))
        self.files = [f for f in all_npy if "_label" not in f]
        if not self.files:
            raise FileNotFoundError(f"在 {data_dir} 找不到 map_*.npy")

        self.idx = 0
        self.cur_lbl = 1  # 默认房间
        self.zoom = 1.0
        self.cx = self.cy = 0.0

        # 每个类别的四边形列表: {label_id: [(x0,y0,x1,y1,x2,y2,x3,y3), ...]}
        self.quads = {1: [], 2: [], 3: [], 4: []}
        
        # 当前正在绘制的四边形
        self._current_points: List[Tuple[int, int]] = []
        self._current_label: int = 1

        # 平移状态
        self._pan = False
        self._pan_ss = (0, 0)
        self._pan_cc = (0.0, 0.0)

        self.occ = None
        self.base = None
        self.H = self.W = 0
        self.path = ""

        self._load(0)
        cv2.namedWindow("QuadAnnotator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("QuadAnnotator", WIN_W, WIN_H)
        cv2.setMouseCallback("QuadAnnotator", self._mouse)

    def _json_path(self, p=None):
        return (p or self.path).replace(".npy", "_quads.json")

    def _label_path(self, p=None):
        return (p or self.path).replace(".npy", "_label.npy")

    def _load(self, idx: int):
        self.idx = idx % len(self.files)
        p = self.files[self.idx]
        self.occ = np.flipud(np.load(p).astype(np.int16))
        self.H, self.W = self.occ.shape
        self.base = occ_to_bgr(self.occ)
        
        # 加载四边形数据
        jp = self._json_path(p)
        if os.path.exists(jp):
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
            self.quads = {1: [], 2: [], 3: [], 4: []}
            for label, items in data.items():
                lid = int(label)
                if lid in self.quads:
                    for item in items:
                        if len(item) == 8:
                            self.quads[lid].append(tuple(item))
            print(f"载入: {os.path.basename(p)} - 房间:{len(self.quads[1])} 走廊:{len(self.quads[2])} 墙壁:{len(self.quads[3])} 其他:{len(self.quads[4])}")
        else:
            self.quads = {1: [], 2: [], 3: [], 4: []}
            print(f"新建: {os.path.basename(p)}")
        
        self._current_points = []
        self.zoom = min(WIN_W / self.W, WIN_H / self.H) * 0.90
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.path = p

    def _save(self):
        """保存为JSON（四边形列表）+ 多通道label"""
        # ① JSON（保留所有四边形，可继续编辑）
        data = {str(k): v for k, v in self.quads.items()}
        with open(self._json_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # ② 多通道label (H, W) - 每种类型独立通道
        # 通道0=房间, 通道1=走廊, 通道2=墙壁, 通道3=其他
        lbl = np.zeros((self.H, self.W), dtype=np.uint8)  # 单通道存储位掩码
        
        for label_id, quad_list in self.quads.items():
            for quad in quad_list:
                x0, y0, x1, y1, x2, y2, x3, y3 = quad
                pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
                mask = np.zeros((self.H, self.W), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 1)
                # 使用位掩码: 1=房间(bit0), 2=走廊(bit1), 4=墙壁(bit2), 8=其他(bit3)
                lbl[mask == 1] |= (1 << (label_id - 1))
        
        np.save(self._label_path(), np.flipud(lbl))
        total = sum(len(v) for v in self.quads.values())
        print(f"✅ {self._json_path()}")
        print(f"✅ {self._label_path()} ({total} 四边形)")

    def _auto_wall(self):
        """插入覆盖全图的墙壁四边形"""
        self.quads[3].append((0, 0, self.W-1, 0, self.W-1, self.H-1, 0, self.H-1))
        print("✅ 已插入'全图墙壁'底图")

    def _s2m(self, sx, sy):
        return (self.cx + (sx - WIN_W / 2) / self.zoom,
                self.cy + (sy - WIN_H / 2) / self.zoom)

    def _m2s(self, mx, my):
        return (int((mx - self.cx) * self.zoom + WIN_W / 2),
                int((my - self.cy) * self.zoom + WIN_H / 2))

    def _clip(self, mx, my):
        return (int(np.clip(mx, 0, self.W - 1)),
                int(np.clip(my, 0, self.H - 1)))

    def _hit(self, sx, sy) -> Tuple[int, int]:
        """返回 (label_id, quad_index) 或 (-1, -1)"""
        mx, my = self._s2m(sx, sy)
        for label_id in [4, 3, 2, 1]:  # 从上到下检查
            for i in range(len(self.quads[label_id]) - 1, -1, -1):
                x0, y0, x1, y1, x2, y2, x3, y3 = self.quads[label_id][i]
                pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
                if cv2.pointPolygonTest(pts, (mx, my), False) >= 0:
                    return label_id, i
        return -1, -1

    def _finish_quad(self):
        if len(self._current_points) >= 3:
            pts = self._current_points[:4]
            while len(pts) < 4:
                pts.append(pts[-1])
            
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x2, y2 = pts[2]
            x3, y3 = pts[3]
            
            self.quads[self._current_label].append((x0, y0, x1, y1, x2, y2, x3, y3))
            n = (LABELS if _PIL else LABELS_EN).get(self._current_label)
            print(f"+ {n}: ({x0},{y0})({x1},{y1})({x2},{y2})({x3},{y3})")
        
        self._current_points = []

    def _render(self) -> np.ndarray:
        canvas = np.full((WIN_H, WIN_W, 3), 20, dtype=np.uint8)
        hw = WIN_W / (2 * self.zoom)
        hh = WIN_H / (2 * self.zoom)
        mx0 = max(0, int(self.cx - hw))
        mx1 = min(self.W, int(self.cx + hw) + 2)
        my0 = max(0, int(self.cy - hh))
        my1 = min(self.H, int(self.cy + hh) + 2)
        if mx0 >= mx1 or my0 >= my1:
            return canvas

        # ① 底图
        crop = self.base[my0:my1, mx0:mx1].copy()
        ov = crop.copy()

        # ② 渲染多标签四边形（每类独立颜色）
        for label_id in [1, 2, 3, 4]:
            color = COLORS[label_id]
            for quad in self.quads[label_id]:
                x0, y0, x1, y1, x2, y2, x3, y3 = quad
                pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
                rx0 = max(0, min(x0, x1, x2, x3) - mx0)
                rx1 = min(mx1 - mx0, max(x0, x1, x2, x3) - mx0 + 1)
                ry0 = max(0, min(y0, y1, y2, y3) - my0)
                ry1 = min(my1 - my0, max(y0, y1, y2, y3) - my0 + 1)
                if rx0 < rx1 and ry0 < ry1:
                    pts_crop = pts - np.array([mx0, my0])
                    cv2.fillPoly(ov, [pts_crop], color)

        crop = cv2.addWeighted(crop, 1 - RECT_ALPHA, ov, RECT_ALPHA, 0)

        # ③ 缩放 & Blit
        sx0 = int((mx0 - self.cx) * self.zoom + WIN_W / 2)
        sy0 = int((my0 - self.cy) * self.zoom + WIN_H / 2)
        sx1 = int((mx1 - self.cx) * self.zoom + WIN_W / 2)
        sy1 = int((my1 - self.cy) * self.zoom + WIN_H / 2)
        sw = max(1, sx1 - sx0)
        sh = max(1, sy1 - sy0)
        interp = cv2.INTER_NEAREST if self.zoom >= 1.0 else cv2.INTER_AREA
        sc = cv2.resize(crop, (sw, sh), interpolation=interp)
        bx = max(0, -sx0); by = max(0, -sy0)
        dx = max(0, sx0); dy = max(0, sy0)
        cw = min(sw - bx, WIN_W - dx)
        ch = min(sh - by, WIN_H - dy)
        if cw > 0 and ch > 0:
            canvas[dy:dy + ch, dx:dx + cw] = sc[by:by + ch, bx:bx + cw]

        # ④ 绘制四边形边框（按层级）
        for label_id in [1, 2, 3, 4]:
            for quad in self.quads[label_id]:
                x0, y0, x1, y1, x2, y2, x3, y3 = quad
                pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
                pts_s = np.array([self._m2s(p[0], p[1]) for p in pts])
                cv2.polylines(canvas, [pts_s], True, COLORS[label_id], 2)

        # ⑤ 当前绘制
        if self._current_points:
            for pt in self._current_points:
                sp = self._m2s(*pt)
                cv2.circle(canvas, sp, 5, COLORS[self._current_label], -1)
            if len(self._current_points) >= 2:
                pts_s = np.array([self._m2s(p[0], p[1]) for p in self._current_points])
                cv2.polylines(canvas, [pts_s], False, COLORS[self._current_label], 2)

        # ⑥ HUD
        cv2.rectangle(canvas, (0, 0), (WIN_W, 30), (30, 30, 30), -1)
        cv2.rectangle(canvas, (WIN_W - 134, 3), (WIN_W - 3, 27),
                      COLORS[self._current_label], -1)
        cv2.rectangle(canvas, (0, WIN_H - 28), (WIN_W, WIN_H), (25, 25, 25), -1)

        # ⑦ 文字
        lbls = LABELS if _PIL else LABELS_EN
        stats = f"房:{len(self.quads[1])} 廊:{len(self.quads[2])} 墙:{len(self.quads[3])} 其:{len(self.quads[4])}"
        texts = [
            (f"[{self.idx+1}/{len(self.files)}] {os.path.basename(self.path)} {stats} zoom:{self.zoom:.2f}x",
             (6, 6), 16, (220, 220, 220)),
            (f"{self._current_label}: {lbls.get(self._current_label, '')}",
             (WIN_W - 130, 8), 14, (255, 255, 255)),
            ("1:房 2:廊 3:墙 4:其他 0:清除  ·  左键:顶点  右键:完成/删除  ·  F:全图墙  Z:撤销  S:保存  N/P:翻页",
             (6, WIN_H - 22), 13, (160, 160, 160)),
        ]
        return draw_texts(canvas, texts)

    def _mouse(self, evt, sx, sy, flags, _):
        if evt == cv2.EVENT_MOUSEWHEEL:
            mx0, my0 = self._s2m(sx, sy)
            factor = 1.15 if flags > 0 else 1.0 / 1.15
            self.zoom = float(np.clip(self.zoom * factor, 0.05, 40.0))
            mx1, my1 = self._s2m(sx, sy)
            self.cx += mx0 - mx1
            self.cy += my0 - my1
            return

        if evt == cv2.EVENT_MBUTTONDOWN:
            self._pan = True
            self._pan_ss = (sx, sy)
            self._pan_cc = (self.cx, self.cy)
        elif evt == cv2.EVENT_MBUTTONUP:
            self._pan = False

        if evt == cv2.EVENT_MOUSEMOVE:
            if self._pan:
                self.cx = self._pan_cc[0] - (sx - self._pan_ss[0]) / self.zoom
                self.cy = self._pan_cc[1] - (sy - self._pan_ss[1]) / self.zoom

        if evt == cv2.EVENT_LBUTTONDOWN:
            pt = self._clip(*self._s2m(sx, sy))
            if len(self._current_points) < 4:
                self._current_points.append(pt)
            if len(self._current_points) == 4:
                self._finish_quad()

        if evt == cv2.EVENT_RBUTTONDOWN:
            if len(self._current_points) >= 3:
                self._finish_quad()
            else:
                label_id, idx = self._hit(sx, sy)
                if label_id >= 1:
                    q = self.quads[label_id].pop(idx)
                    n = (LABELS if _PIL else LABELS_EN).get(label_id)
                    print(f"- 删除 {n} 四边形")

    def run(self):
        while True:
            cv2.imshow("QuadAnnotator", self._render())
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k == 27:
                self._save()
                break
            if k == ord('q'):
                self._save()
                break
            elif k == ord('s'):
                self._save()
            elif k == ord('z'):
                if self._current_points:
                    self._current_points.pop()
                elif any(len(v) > 0 for v in self.quads.values()):
                    for lid in [4, 3, 2, 1]:
                        if self.quads[lid]:
                            q = self.quads[lid].pop()
                            n = (LABELS if _PIL else LABELS_EN).get(lid)
                            print(f"撤销: {n}")
                            break
            elif k == ord('f'):
                self._auto_wall()
            elif k == ord('c'):
                self.quads = {1: [], 2: [], 3: [], 4: []}
                self._current_points = []
                print("清空所有")
            elif k == ord('\n') or k == ord('\r'):
                self._finish_quad()
            elif k == ord('n'):
                self._save()
                self._load(self.idx + 1)
            elif k == ord('p'):
                self._save()
                self._load(self.idx - 1)
            elif k == ord('0'):
                self.cur_lbl = 0
                print("标签: 清除模式（右键删除）")
            elif ord('1') <= k <= ord('4'):
                self._current_label = k - ord('0')
                self.cur_lbl = self._current_label
                n = (LABELS if _PIL else LABELS_EN).get(self._current_label)
                print(f"标签: {self._current_label} - {n}")
        
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="多标签四边形标注工具")
    ap.add_argument("--data_dir", default="map_data")
    args, _ = ap.parse_known_args()
    QuadAnnotator(args.data_dir).run()


if __name__ == "__main__":
    main()
