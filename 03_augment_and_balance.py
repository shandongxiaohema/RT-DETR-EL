# python
import os
import cv2
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm
import albumentations as A
import random
import numpy as np

# ==================== 路径配置 ====================
img_dir = r"E:\edgedownload\mdoels\实验\EL2021_split\pavel_split\train-原始\JPEGImages"
xml_dir = r"E:\edgedownload\mdoels\实验\EL2021_split\pavel_split\train-原始\Annotations"

save_root = r"E:\edgedownload\mdoels\实验\EL2021_split\pavel_split\pavel_v4\train"
save_img_dir = os.path.join(save_root, "JPEGImages")
save_xml_dir = os.path.join(save_root, "Annotations")
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_xml_dir, exist_ok=True)

# ==================== 参数 ====================
TARGET_CLASSES = ['thick_line', 'horizontal_dislocation', 'black_core', 'finger', 'short_circuit', 'crack']
MIN_BOX_W = 5
MIN_BOX_H = 5
MAX_AUG_TRIES = 6
MAX_PER_IMAGE = 8
SEED = 42

# 抖动目标数设置：基准值 ± 抖动比例
BASE_TARGET = 1500          # 基准目标数
TARGET_JITTER_FRAC = 0.10   # 例如 0.10 表示 ±10%
DOWNSAMPLE_MAJOR = True     # 多数类是否下采样到各自目标

random.seed(SEED)
np.random.seed(SEED)

# 仅做水平/垂直翻转 + 亮度对比度 + 高斯噪声
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.6),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.0))

# ==================== 工具函数 ====================
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    bboxes, labels = [], []
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip()
        if name not in TARGET_CLASSES:
            continue
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text); ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text); ymax = float(bbox.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
    return filename, bboxes, labels, w, h

def clamp_box(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(0, min(xmax, w - 1))
    ymax = max(0, min(ymax, h - 1))
    return [xmin, ymin, xmax, ymax]

def is_valid_box(box, w, h):
    xmin, ymin, xmax, ymax = clamp_box(box, w, h)
    if xmin >= xmax or ymin >= ymax:
        return False
    if (xmax - xmin) < MIN_BOX_W or (ymax - ymin) < MIN_BOX_H:
        return False
    return True

def save_annotation_xml(out_xml_path, jpg_name, boxes, labels, w, h):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "augmented"
    ET.SubElement(annotation, "filename").text = jpg_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = map(lambda x: str(int(round(x))), box)
        ET.SubElement(bnd, "xmin").text = xmin
        ET.SubElement(bnd, "ymin").text = ymin
        ET.SubElement(bnd, "xmax").text = xmax
        ET.SubElement(bnd, "ymax").text = ymax
    ET.ElementTree(annotation).write(out_xml_path, encoding="utf-8")

def rebuild_counts(records):
    counts = defaultdict(int)
    file_index = defaultdict(list)
    for idx, rec in enumerate(records):
        if not rec['labels']:
            continue
        for l in rec['labels']:
            counts[l] += 1
        for cls in set(rec['labels']):
            file_index[cls].append(idx)
    return counts, file_index

def index_occurrences(records, cls):
    occ = []
    for i, rec in enumerate(records):
        for j, l in enumerate(rec['labels']):
            if l == cls:
                occ.append((i, j))
    return occ

def remove_occurrences(records, occ_to_remove):
    per_rec = defaultdict(list)
    for i, j in occ_to_remove:
        per_rec[i].append(j)
    for i, js in per_rec.items():
        for j in sorted(js, reverse=True):
            del records[i]['bboxes'][j]
            del records[i]['labels'][j]

def build_targets(class_counts):
    base = BASE_TARGET if (BASE_TARGET and BASE_TARGET > 0) else max(class_counts.get(c, 0) for c in TARGET_CLASSES) or 1
    lo = int(round(base * (1 - TARGET_JITTER_FRAC)))
    hi = int(round(base * (1 + TARGET_JITTER_FRAC)))
    targets = {}
    for cls in TARGET_CLASSES:
        tgt = random.randint(lo, hi)
        if not DOWNSAMPLE_MAJOR:
            tgt = max(tgt, class_counts.get(cls, 0))
        targets[cls] = tgt
    return targets

# ==================== 读取原始数据 ====================
records = []
xml_files = [f for f in os.listdir(xml_dir) if f.lower().endswith(".xml")]
for xml_file in tqdm(xml_files, desc="读取原始 XML"):
    xml_path = os.path.join(xml_dir, xml_file)
    filename, bboxes, labels, w, h = parse_xml(xml_path)
    if not bboxes:
        continue
    records.append({
        "xml_file": xml_file, "filename": filename,
        "bboxes": bboxes, "labels": labels, "w": w, "h": h
    })

class_counts, class_to_files = rebuild_counts(records)
print("原始六类数量:", {c: class_counts.get(c, 0) for c in TARGET_CLASSES})

# ==================== 生成各类“抖动目标数”并下采样多数类 ====================
targets = build_targets(class_counts)
print("各类抖动目标数:", targets)

if DOWNSAMPLE_MAJOR:
    for cls in TARGET_CLASSES:
        cur = class_counts.get(cls, 0)
        if cur > targets[cls]:
            need_remove = cur - targets[cls]
            occ = index_occurrences(records, cls)
            if need_remove > 0 and occ:
                to_remove = set(random.sample(occ, min(need_remove, len(occ))))
                remove_occurrences(records, to_remove)
    # 清理无标注样本
    before = len(records)
    records = [r for r in records if len(r['bboxes']) > 0]
    if before - len(records) > 0:
        print(f"清理空样本: 移除 {before - len(records)} 条")
    class_counts, class_to_files = rebuild_counts(records)
    print("下采样后六类数量:", {c: class_counts.get(c, 0) for c in TARGET_CLASSES})

# ==================== 拷贝（下采样后的）原始数据 ====================
global_idx = 1
for rec in tqdm(records, desc="拷贝原始数据"):
    if not rec['bboxes']:
        continue
    src_img = os.path.join(img_dir, rec["filename"])
    img = cv2.imread(src_img)
    if img is None:
        continue
    h, w = img.shape[:2]
    out_img = f"{global_idx}.jpg"
    out_xml = f"{global_idx}.xml"
    cv2.imwrite(os.path.join(save_img_dir, out_img), img)
    save_annotation_xml(os.path.join(save_xml_dir, out_xml), out_img, rec["bboxes"], rec["labels"], w, h)
    global_idx += 1

# ==================== 增强少数类到各自“抖动目标数” ====================
per_image_aug_used = defaultdict(int)
total_aug = 0

def need_more(counts, targets_dict):
    return {c: max(0, targets_dict[c] - counts.get(c, 0)) for c in TARGET_CLASSES}

needs = need_more(class_counts, targets)
print("增强前缺口:", needs)

for cls in TARGET_CLASSES:
    if needs[cls] <= 0:
        continue
    indices = class_to_files.get(cls, [])
    if not indices:
        continue
    random.shuffle(indices)
    for idx in indices:
        if needs[cls] <= 0:
            break
        rec = records[idx]
        if per_image_aug_used[idx] >= MAX_PER_IMAGE:
            continue
        img_path = os.path.join(img_dir, rec["filename"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        tries_left = MAX_PER_IMAGE - per_image_aug_used[idx]
        for _ in range(tries_left):
            if needs[cls] <= 0:
                break
            success = False
            for _ in range(MAX_AUG_TRIES):
                augmented = augmenter(image=img, bboxes=rec['bboxes'], category_ids=rec['labels'])
                aug_img = augmented['image']
                aug_boxes = list(augmented['bboxes'])
                aug_labels = list(augmented['category_ids'])

                # 仅保留目标类别，避免放大其他类
                kept_boxes, kept_labels = [], []
                for b, l in zip(aug_boxes, aug_labels):
                    if l != cls:
                        continue
                    if is_valid_box(b, w, h):
                        kept_boxes.append(clamp_box(b, w, h))
                        kept_labels.append(l)

                if not kept_boxes:
                    continue

                out_img = f"{global_idx}.jpg"
                out_xml = f"{global_idx}.xml"
                cv2.imwrite(os.path.join(save_img_dir, out_img), aug_img)
                save_annotation_xml(os.path.join(save_xml_dir, out_xml), out_img, kept_boxes, kept_labels, w, h)

                add_cnt = sum(1 for l in kept_labels if l == cls)
                class_counts[cls] += add_cnt
                needs = need_more(class_counts, targets)
                per_image_aug_used[idx] += 1
                global_idx += 1
                total_aug += 1
                success = True
                break
            if not success:
                continue

print("\n✅ 增强完成")
print("最终六类数量(估计):", {c: class_counts.get(c, 0) for c in TARGET_CLASSES})
print(f"增强样本数: {total_aug}")
print(f"输出路径: {save_root}")
