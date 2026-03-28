#!/usr/bin/env python3
"""
03_train.py — 真正的训练/验证分离：用不同的地图
类别：房间、走廊、墙壁、其他（背景=全0）
"""
import os, glob, argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ==============================================================================
# 配置
# ==============================================================================
NUM_CLASSES = 4
LABEL_NAMES = ["房间", "走廊", "墙壁", "其他"]

LABEL_COLORS = np.array([
    [0, 200, 0],        # 0 房间
    [0, 220, 220],      # 1 走廊
    [60, 60, 220],      # 2 墙壁
    [0, 140, 255],      # 3 其他
    [100, 100, 100]     # 背景（仅用于可视化）
], dtype=np.uint8)

# ==============================================================================
# 模型
# ==============================================================================
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
    def __init__(self, c_in=1, num_classes=4, base=32):
        super().__init__()
        b = base
        self.e1 = DoubleConv(c_in, b)
        self.e2 = Down(b, b*2)
        self.e3 = Down(b*2, b*4)
        self.e4 = Down(b*4, b*8)
        self.bot = DoubleConv(b*8, b*8)
        self.d3 = Up(b*8, b*4, b*4)
        self.d2 = Up(b*4, b*2, b*2)
        self.d1 = Up(b*2, b, b)
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

# ==============================================================================
# 数据处理
# ==============================================================================
def norm_occ(occ):
    out = np.zeros_like(occ, np.float32)
    valid = occ >= 0
    out[valid] = occ[valid] / 100.0
    return out

def lbl_to_4channel(lbl):
    h, w = lbl.shape
    out = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    out[0] = ((lbl & 1) != 0)
    out[1] = ((lbl & 2) != 0)
    out[2] = ((lbl & 4) != 0)
    out[3] = ((lbl & 8) != 0)
    return out

def random_crop(img, target, size):
    """同时裁剪图像和目标，确保同一位置！"""
    h, w = img.shape
    # 先pad
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    h, w = img.shape
    # 随机选择裁剪位置
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    # 裁剪
    img_crop = img[y:y+size, x:x+size]
    target_crop = target[:, y:y+size, x:x+size]
    return img_crop, target_crop

class MapDataset(Dataset):
    def __init__(self, pairs, crop=256, aug=True):
        self.pairs = pairs
        self.crop = crop
        self.aug = aug
        self.data = []
        for occ_f, lbl_f in pairs:
            occ = np.flipud(np.load(occ_f)).astype(np.int16)
            lbl = np.flipud(np.load(lbl_f)).astype(np.uint8)
            self.data.append((occ, lbl))

    def __len__(self):
        return len(self.data) * 100 if self.aug else len(self.data)

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        occ, lbl = self.data[idx]

        img = norm_occ(occ)
        target = lbl_to_4channel(lbl)

        if self.crop:
            img, target = random_crop(img, target, self.crop)

        if self.aug:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                target = np.flip(target, axis=-1).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                target = np.flip(target, axis=-2).copy()

        img = torch.from_numpy(img.copy())[None].float()
        target = torch.from_numpy(target.copy()).float()
        return img, target

def collect_all_data(base_dir="map_data _all"):
    """自动从base_dir下收集所有子目录的数据"""
    all_pairs = []
    if not os.path.isdir(base_dir):
        print(f"警告: {base_dir} 目录不存在")
        return all_pairs

    # 如果base_dir下直接有map文件，也收集
    fs = sorted([f for f in glob.glob(os.path.join(base_dir, "map_*.npy")) if "_label" not in f])
    if fs:
        pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
        all_pairs.extend(pairs)
        print(f"从 {base_dir} 加载了 {len(pairs)} 张地图")

    # 收集所有子目录，按编号排序 map_data_1, map_data_2 ...
    subdirs = []
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir) and entry.startswith("map_data_"):
            try:
                idx = int(entry.split("_")[-1])
                subdirs.append((idx, entry, subdir))
            except:
                continue
    # 按编号从小到大排序
    subdirs.sort(key=lambda x: x[0])
    for idx, entry, subdir in subdirs:
        fs = sorted([f for f in glob.glob(os.path.join(subdir, "map_*.npy")) if "_label" not in f])
        if fs:
            pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
            all_pairs.extend(pairs)
            print(f"从 {subdir} 加载了 {len(pairs)} 张地图")
    return all_pairs

def collect_pairs_from_dirs(dirs):
    """从多个目录收集数据对"""
    all_pairs = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        fs = sorted([f for f in glob.glob(os.path.join(d, "map_*.npy")) if "_label" not in f])
        pairs = [(f, f.replace(".npy", "_label.npy")) for f in fs if os.path.exists(f.replace(".npy", "_label.npy"))]
        all_pairs.extend(pairs)
        print(f"从 {d} 加载了 {len(pairs)} 张地图")
    return all_pairs

# ==============================================================================
# 评估 & 可视化
# ==============================================================================
@torch.no_grad()
def compute_iou(pred, target, th=0.5):
    pred = (torch.sigmoid(pred) > th).float()
    ious = []
    for c in range(NUM_CLASSES):
        p = pred[:, c:c+1]
        t = target[:, c:c+1]
        inter = (p * t).sum()
        union = (p + t).clamp(0, 1).sum()
        if union > 0:
            iou = (inter / (union + 1e-8)).item()
        else:
            iou = float('nan')
        ious.append(iou)
    return ious

@torch.no_grad()
def compute_accuracy(pred, target, th=0.5):
    """计算像素准确率：正确预测的像素比例"""
    pred_bin = (torch.sigmoid(pred) > th)
    correct = (pred_bin == target).sum().item()
    total = target.numel()
    return correct / total

def colorize(lbl):
    c, h, w = lbl.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    has_label = np.zeros((h, w), dtype=bool)
    for cls in range(NUM_CLASSES):
        mask = (lbl[cls] == 1)
        has_label |= mask
        if np.any(mask):
            out[mask] = np.clip(out[mask].astype(np.float32) * 0.5 + LABEL_COLORS[cls] * 0.5, 0, 255).astype(np.uint8)
    bg_mask = ~has_label
    out[bg_mask] = LABEL_COLORS[4]
    return out

@torch.no_grad()
def save_samples(model, ds, device, out_dir, n=4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for i in range(min(n, len(ds))):
        img, tgt = ds[i]
        logits = model(img[None].to(device))
        pred = (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 4))
        a1.imshow(img[0].numpy(), cmap='gray')
        a2.imshow(colorize(tgt.numpy()))
        a3.imshow(colorize(pred))
        a1.set_title('Input')
        a2.set_title('Label')
        a3.set_title('Pred')
        for a in [a1, a2, a3]:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'sample_{i}.png'), dpi=100)
        plt.close()

# ==============================================================================
# 训练
# ==============================================================================
def train(args):
    # 收集所有数据
    if args.data_dirs.strip():
        # 用户指定了目录
        data_dirs = [d.strip() for d in args.data_dirs.split(',')]
        all_pairs = collect_pairs_from_dirs(data_dirs)
    else:
        # 自动发现 map_data _all 下所有数据
        all_pairs = collect_all_data("map_data _all")

    if not all_pairs:
        raise Exception(f"未找到数据，请检查目录: map_data _all/")

    if not all_pairs:
        raise Exception(f"未找到数据，请检查目录: {args.data_dirs}")

    print(f"总共 {len(all_pairs)} 张地图，将按顺序增量训练")
    print("="*80)

    device = torch.device(args.device)
    model = UNet(num_classes=4, base=args.base_ch).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    total_epoch = 0
    best_miou_global = 0.0

    # 是否从已有模型继续训练
    if args.resume.strip() and os.path.exists(args.resume):
        print(f"\n🔄 从已有模型继续训练: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt)
        print(f"✅ 模型加载完成，继续训练\n")
    elif args.resume.strip():
        print(f"\n⚠️  模型文件不存在: {args.resume}，从头开始训练\n")

    # 多轮训练：每轮都按顺序 1→N 遍历所有地图
    for round_idx in range(1, args.max_rounds+1):
        print(f"\n\n{'#'*80}")
        print(f"🔄 第 {round_idx}/{args.max_rounds} 轮训练开始")
        print(f"{'#'*80}\n")

        # 按顺序依次训练每一张地图
        for map_idx, (current_pair) in enumerate(all_pairs):
            # 当前这张地图，自己做训练和验证
            train_p = [current_pair]
            val_p = [current_pair]
            print(f"\n\n{'='*80}")
            # 当前轮次的停止阈值：从0.3梯度上升到0.98
            # 0.8 之前每轮 +0.1，达到 0.8 之后放缓到 +0.03 每轮
            if round_idx <= 6:
                current_stop_miou = 0.3 + (round_idx - 1) * 0.1
            else:
                current_stop_miou = min(0.8 + (round_idx - 6) * 0.03, 0.98)
            print(f"▶️  开始训练第 {map_idx+1}/{len(all_pairs)} 张 (轮 {round_idx}): {os.path.basename(current_pair[0])}")
            print(f"    训练集: 1 张 (当前地图), 验证集: 1 张 (当前地图)")
            print(f"    停止条件: 验证 mIoU > {current_stop_miou:.1%} 即进入下一张 (梯度递增)")
            print("-"*80)

            # 数据集
            ds_train = MapDataset(train_p, crop=args.crop_size, aug=True)
            ds_val   = MapDataset(val_p,   crop=None, aug=False)
            dl_train = DataLoader(ds_train, args.batch_size, shuffle=True, num_workers=0)
            dl_val   = DataLoader(ds_val,   1, shuffle=False, num_workers=0)

            best_miou = 0.0
            best_epoch = 0
            no_improve = 0
            patience = args.patience

            if round_idx == 1 and map_idx == 0:
                print("\nEpoch │ Train │  Val  │ Acc  │ mIoU │ 房间 │ 走廊 │ 墙壁 │ 其他")
                print("─"*90)

            for ep in range(1, args.max_epochs_per_map+1):
                total_epoch += 1
                model.train()
                t_loss = 0.0
                for img, tgt in dl_train:
                    img, tgt = img.to(device), tgt.to(device)
                    opt.zero_grad()
                    logits = model(img)
                    loss = criterion(logits, tgt)
                    loss.backward()
                    opt.step()
                    t_loss += loss.item()

                # 验证集（整图）
                model.eval()
                v_loss = 0.0
                iou_sum = [0.0]*4
                cnt = [0]*4
                acc_sum = 0.0
                acc_cnt = 0
                with torch.no_grad():
                    for img, tgt in dl_val:
                        img, tgt = img.to(device), tgt.to(device)
                        logits = model(img)
                        loss = criterion(logits, tgt)
                        v_loss += loss.item()
                        ious = compute_iou(logits, tgt)
                        acc = compute_accuracy(logits, tgt)
                        acc_sum += acc
                        acc_cnt += 1
                        for c, v in enumerate(ious):
                            if not np.isnan(v):
                                iou_sum[c] += v
                                cnt[c] += 1

                ious = [iou_sum[c]/max(cnt[c], 1) for c in range(4)]
                miou = float(np.nanmean(ious))
                accuracy = acc_sum / max(acc_cnt, 1)
                avg_tloss = t_loss / len(dl_train)
                avg_vloss = v_loss / len(dl_val)

                # 保存最好的模型
                improved = False
                if miou > best_miou:
                    best_miou = miou
                    if miou > best_miou_global:
                        best_miou_global = miou
                    best_epoch = total_epoch
                    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best.pth"))
                    save_samples(model, ds_val, device, args.ckpt_dir)
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1

                # 打印
                if improved:
                    print(f"{total_epoch:4d} │ {avg_tloss:6.3f} │ {avg_vloss:6.3f} │ {accuracy:.3f} │ {miou:.3f} │ "
                          f"{ious[0]:.3f} │ {ious[1]:.3f} │ {ious[2]:.3f} │ {ious[3]:.3f} │ ✅")
                else:
                    print(f"{total_epoch:4d} │ {avg_tloss:6.3f} │ {avg_vloss:6.3f} │ {accuracy:.3f} │ {miou:.3f} │ "
                          f"{ious[0]:.3f} │ {ious[1]:.3f} │ {ious[2]:.3f} │ {ious[3]:.3f}")

                # 检查停止条件：当前轮次mIoU达到阈值就下一张
                if miou >= current_stop_miou:
                    print(f"\n✅ 第 {map_idx+1} 张训练完成 (轮 {round_idx}), 验证 mIoU {miou:.1%} ≥ {current_stop_miou:.1%}，进入下一张")
                    break

                if no_improve >= patience:
                    print(f"\n⚠️  {patience} 个epoch没提升，提前进入下一张")
                    break

            # mIoU > 0.999 全局停止
            if best_miou_global > 0.999:
                print(f"\n🎉 全局验证 mIoU = {best_miou_global:.4f} > 0.999，提前停止全部训练！")
                break

        # 一轮完成
        if best_miou_global > 0.999:
            break

    print(f"\n{'='*80}")
    print(f"✅ 全部 {args.max_rounds} 轮训练完成！")
    print(f"最好模型保存在 {args.ckpt_dir}/best.pth")
    print(f"全局最好 mIoU: {best_miou_global:.4f}")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs",
                    default="",
                    help="数据目录，多个用逗号分隔，留空则自动发现 map_data _all 下所有数据")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--stop_miou", type=float, default=0.5,
                    help="每张地图达到此mIoU后进入下一张")
    ap.add_argument("--max_epochs_per_map", type=int, default=100,
                    help="每张地图最大训练epoch数")
    ap.add_argument("--max_rounds", type=int, default=10,
                    help="最多训练多少轮，每轮遍历所有地图")
    ap.add_argument("--patience", type=int, default=30,
                    help="早停耐心值，达到则进入下一张")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", nargs='?', const="checkpoints/best.pth", default="",
                    help="从已有模型继续训练。--resume (默认 checkpoints/best.pth) 或 --resume /path/to/model；留空则从头训练")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()
