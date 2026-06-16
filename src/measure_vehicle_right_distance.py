from pathlib import Path
from contextlib import contextmanager
import json
import csv

import cv2
import numpy as np
from ultralytics import YOLO
import project_config as cfg


# =========================
# 文件配置
# =========================
VIDEO_NAME = cfg.VIDEO_NAME
LANE_MAP_NAME = cfg.LANE_MAP_NAME
MODEL_NAME = cfg.MODEL_NAME
TRAFFIC_DIRECTION = str(getattr(cfg, "TRAFFIC_DIRECTION", "AWAY_FROM_CAMERA")).upper()

START_FRAME = 0
MAX_FRAMES = 800

# =========================
# 显示控制
# =========================
SHOW_DEBUG_TEXT = False
SHOW_VEHICLE_BOX = True
SHOW_VEHICLE_MASK_EDGE = False
DRAW_ROI_BORDER = False
SHOW_DISTANCE_TEXT = True
SHOW_DISTANCE_LINE = True

# =========================
# 车道线透明度
# =========================
CONTINUOUS_ALPHA = 0.50
PAINT_ALPHA = 0.75
GHOST_ALPHA = 0.12

# =========================
# 线宽
# =========================
CONTINUOUS_THICKNESS = 3
PAINT_THICKNESS = 4
GHOST_THICKNESS = 1

# =========================
# 软遮挡参数
# =========================
SOFT_MASK_BLUR_KSIZE = 17
NORMAL_MASK_BLUR_KSIZE = 3
GHOST_MASK_BLUR_KSIZE = 5

VEHICLE_MASK_DILATE_X = 15
VEHICLE_MASK_DILATE_Y = 9

# =========================
# YOLO + 自定义跟踪参数
# =========================
# YOLO 模型入口置信度：降低入口阈值，先让远处小目标摩托车进入候选。
# 后续再按车型二次过滤，避免汽车/货车/客车误检明显增加。
YOLO_CONF = float(getattr(cfg, "YOLO_CONF", 0.08))
YOLO_IOU = float(getattr(cfg, "YOLO_IOU", 0.45))
YOLO_IMGSZ = 1536

# 分车型二次置信度阈值
CAR_MIN_CONF = float(getattr(cfg, "CAR_MIN_CONF", 0.18))
TRUCK_MIN_CONF = float(getattr(cfg, "TRUCK_MIN_CONF", 0.18))
BUS_MIN_CONF = float(getattr(cfg, "BUS_MIN_CONF", 0.18))
MOTORCYCLE_MIN_CONF = float(getattr(cfg, "MOTORCYCLE_MIN_CONF", 0.08))

CUSTOM_TRACK_MAX_MISSING = 8
CUSTOM_TRACK_MIN_HITS = 2
CUSTOM_BOX_SMOOTH_ALPHA = 0.45
CUSTOM_MATCH_IOU_TH = 0.05
CUSTOM_MATCH_CENTER_RATIO = 1.15

DRAW_HOLD_BOX_MAX_MISSING = 0
NMS_IOU_TH = 0.22
MIN_ROI_INTERSECTION = 20

MASK_BOX_MAX_AREA_RATIO = 1.45
MASK_BOX_MIN_AREA_RATIO = 0.20
MASK_BOX_MAX_CENTER_SHIFT_RATIO = 0.45

TRACK_DUP_IOU_TH = 0.18
TRACK_DUP_CENTER_RATIO = 0.72
TRACK_DUP_AREA_RATIO_MIN = 0.25
TRACK_DUP_AREA_RATIO_MAX = 3.60
TRACK_DUP_CONTAINMENT_TH = 0.55

# 新建 ID 前查重：防止同一辆车被拆成两个 track。
NEW_TRACK_DUP_IOU_TH = 0.08
NEW_TRACK_DUP_CENTER_RATIO = 0.82
NEW_TRACK_DUP_CONTAINMENT_TH = 0.50

# 最终显示前再查重：保证同一帧同一辆车只显示一个 ID。
FINAL_BOX_DUP_IOU_TH = 0.16
FINAL_BOX_DUP_CENTER_RATIO = 0.78
FINAL_BOX_DUP_CONTAINMENT_TH = 0.48

# 同一帧 YOLO 重复检测抑制。
NMS_CENTER_RATIO = 0.50
NMS_CONTAINMENT_TH = 0.55


VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}

# =========================
# 距离计算参数
# =========================
LANE_WIDTH_M = float(getattr(cfg, "LANE_WIDTH_M", 3.75))

# 车辆右侧边界取车辆底部附近的横截面
REF_Y_FROM_BOTTOM_RATIO = 0.12
RIGHT_EDGE_BAND_RATIO = 0.12
RIGHT_EDGE_BAND_MIN = 6
RIGHT_EDGE_BAND_MAX = 28

# 状态阈值，单位 m
NEAR_DIST_M = 0.60
TOUCH_DIST_M = 0.15
CROSS_DIST_M = -0.10

# 过滤极端异常值
MAX_VALID_DISTANCE_M = 8.0

# 右边缘鲁棒提取：避免 mask 单点毛刺把右边界拉偏
RIGHT_EDGE_ROW_MIN_PIXELS = 4
RIGHT_EDGE_MIN_VALID_ROWS = 3
RIGHT_EDGE_PERCENTILE = 90

# v4.2：不再只取车辆底部单一横截面，而是在车身侧面范围内扫描
# 这样可避免测距点落在车尾/车头的窄小端部。
RIGHT_EDGE_SCAN_TOP_RATIO = 0.20
RIGHT_EDGE_SCAN_BOTTOM_RATIO = 0.96
RIGHT_EDGE_ROW_WIDTH_RATIO_MIN = 0.12
RIGHT_EDGE_DISTANCE_PERCENTILE = 12

# v4.3：车道归属与右侧测距点分离
# 车道归属不再使用 bbox 中心，而是使用车辆底部可见轮廓中心。
LANE_ASSIGN_SCAN_TOP_RATIO = 0.45
LANE_ASSIGN_SCAN_BOTTOM_RATIO = 0.92
LANE_ASSIGN_WIDTH_KEEP_RATIO = 0.35
LANE_ASSIGN_Y_PERCENTILE = 76

# 右边缘不再选“最接近右线”的单行，而是选车辆下半部稳定侧边轮廓。
# 这样可减少车头/车尾窄边、后视镜、mask 毛刺造成的负距离误判。
RIGHT_EDGE_FOOTPRINT_TOP_RATIO = 0.40
RIGHT_EDGE_FOOTPRINT_BOTTOM_RATIO = 0.90
RIGHT_EDGE_FOOTPRINT_WIDTH_KEEP_RATIO = 0.45
RIGHT_EDGE_FOOTPRINT_Y_PERCENTILE = 72
RIGHT_EDGE_RIGHT_X_PERCENTILE = 75
RIGHT_EDGE_LOCAL_Y_WINDOW_RATIO = 0.08

# 时序平滑与车道归属滞回
RIGHT_EDGE_SMOOTH_ALPHA = 0.50
DISTANCE_SMOOTH_ALPHA = 0.38
LANE_ASSIGN_HYSTERESIS_RATIO = 0.16
LANE_ASSIGN_CONFIRM_FRAMES = 3

# 换道判断阈值：v4.3 增加连续帧确认和释放滞回
DEPARTURE_CONFIRM_FRAMES = 4
DEPARTURE_RELEASE_CONFIRM_FRAMES = 5
DEPARTURE_RELEASE_DIST_M = 0.22
LANE_CHANGE_CONFIRM_FRAMES = 8
LANE_CHANGE_HOLD_FRAMES = 18
RIGHT_APPROACH_SPEED_MPS = 0.15
LANE_CHANGE_CENTER_RATIO_TH = 1.03

# v4.4：修正车道切换后的平滑拖尾。
# 当车辆从原车道切换到相邻车道后，距离基准线已经改变，必须重置距离平滑，
# 否则会把旧车道的负距离延续到新车道，造成 CROSS/RIGHT_DEPARTURE 假象。
RESET_SMOOTH_ON_LANE_PAIR_CHANGE = True
LANE_PAIR_RESET_DISTANCE_JUMP_M = 1.20

# v4.4：RIGHT_DEPARTURE 只表示已经连续越过右线，不再把 TOUCH+靠近速度直接判为偏离。
# TOUCH/NEAR 仍由 status 字段表达。
DEPARTURE_USE_CROSS_ONLY = True

# v4.5：右边缘时序异常跳变约束。
# YOLO-seg mask 偶尔会和阴影/相邻目标粘连，导致右边缘点突然外跳；
# 这里用上一帧右边缘和 bbox 中心位移预测本帧允许范围，超出范围时截断。
EDGE_SANITY_ENABLE = True

# v4.9：v4.5 的允许跳变量过大，ID3 在 307/308 帧仍可出现 150px 级别外跳。
# 改为“最小允许跳变 + bbox 宽度比例 + 硬上限”的组合，避免大车 bbox 过宽时放开约束。
EDGE_SANITY_MIN_ABS_JUMP_PX = 28
EDGE_SANITY_MAX_ABS_JUMP_PX = 58
EDGE_SANITY_BBOX_WIDTH_RATIO = 0.14
EDGE_SANITY_BBOX_SHIFT_GAIN = 1.00

# v4.9：右换道后锁定车道对，防止分割 mask 中心短时回跳导致测距基准线退回旧车道。
LANE_PAIR_LOCK_AFTER_RIGHT_CHANGE_FRAMES = 35
LANE_PAIR_LOCK_ALLOW_LEFT_REVERT_RATIO = -0.45
LANE_PAIR_LOCK_LEFT_REVERT_CONFIRM_FRAMES = 8

# v4.9：测距质量门控。
# 近景大车/半出画车辆的 bbox 和 mask 透视畸变很强，不能可靠判断右侧距离和换道。
# 这类目标保留检测框，但不输出距离线，不参与 RIGHT_DEPARTURE / RIGHT_LANE_CHANGE。
MEASUREMENT_QUALITY_GATE_ENABLE = True
MEASUREMENT_BOTTOM_CLIP_MARGIN_PX = 16
MEASUREMENT_SIDE_CLIP_MARGIN_PX = 4
MEASUREMENT_MAX_BBOX_AREA_RATIO = 0.16
MEASUREMENT_MAX_BBOX_HEIGHT_RATIO = 0.50
MEASUREMENT_MAX_BBOX_WIDTH_RATIO = 0.62
MEASUREMENT_MAX_BBOX_WIDTH_LANE_RATIO = 2.20
MEASUREMENT_MIN_VALID_LANE_WIDTH_PX = 35.0
MEASUREMENT_LOW_REF_MARGIN_PX = 22

# =========================
# 车型差异化参数
# =========================
# 普通汽车/货车/客车继续使用上面的全局参数；摩托车体积小、mask 行数少，
# 因此使用更低的有效像素/有效行阈值、更宽松的 ROI 交集阈值，以及更严格的 bbox 尺寸门控。
MOTORCYCLE_CUSTOM_TRACK_MAX_MISSING = 10
MOTORCYCLE_CUSTOM_TRACK_MIN_HITS = 1
MOTORCYCLE_MIN_ROI_INTERSECTION = 8

MOTORCYCLE_MASK_BOX_MAX_AREA_RATIO = 1.80
MOTORCYCLE_MASK_BOX_MIN_AREA_RATIO = 0.10
MOTORCYCLE_MASK_BOX_MAX_CENTER_SHIFT_RATIO = 0.60

MOTORCYCLE_RIGHT_EDGE_ROW_MIN_PIXELS = 2
MOTORCYCLE_RIGHT_EDGE_MIN_VALID_ROWS = 2
MOTORCYCLE_RIGHT_EDGE_BAND_MIN = 3
MOTORCYCLE_RIGHT_EDGE_BAND_MAX = 20

MOTORCYCLE_LANE_ASSIGN_SCAN_TOP_RATIO = 0.35
MOTORCYCLE_LANE_ASSIGN_SCAN_BOTTOM_RATIO = 0.95
MOTORCYCLE_LANE_ASSIGN_WIDTH_KEEP_RATIO = 0.22
MOTORCYCLE_LANE_ASSIGN_Y_PERCENTILE = 78
MOTORCYCLE_LANE_ASSIGN_CONFIRM_FRAMES = 2

MOTORCYCLE_RIGHT_EDGE_FOOTPRINT_TOP_RATIO = 0.30
MOTORCYCLE_RIGHT_EDGE_FOOTPRINT_BOTTOM_RATIO = 0.95
MOTORCYCLE_RIGHT_EDGE_FOOTPRINT_WIDTH_KEEP_RATIO = 0.25
MOTORCYCLE_RIGHT_EDGE_FOOTPRINT_Y_PERCENTILE = 76
MOTORCYCLE_RIGHT_EDGE_RIGHT_X_PERCENTILE = 70
MOTORCYCLE_RIGHT_EDGE_LOCAL_Y_WINDOW_RATIO = 0.12

MOTORCYCLE_EDGE_SANITY_MIN_ABS_JUMP_PX = 18
MOTORCYCLE_EDGE_SANITY_MAX_ABS_JUMP_PX = 42
MOTORCYCLE_EDGE_SANITY_BBOX_WIDTH_RATIO = 0.22

MOTORCYCLE_MEASUREMENT_BOTTOM_CLIP_MARGIN_PX = 10
MOTORCYCLE_MEASUREMENT_SIDE_CLIP_MARGIN_PX = 2
MOTORCYCLE_MEASUREMENT_MAX_BBOX_AREA_RATIO = 0.08
MOTORCYCLE_MEASUREMENT_MAX_BBOX_HEIGHT_RATIO = 0.38
MOTORCYCLE_MEASUREMENT_MAX_BBOX_WIDTH_RATIO = 0.28
MOTORCYCLE_MEASUREMENT_MAX_BBOX_WIDTH_LANE_RATIO = 1.20
MOTORCYCLE_MEASUREMENT_LOW_REF_MARGIN_PX = 12

MOTORCYCLE_DEPARTURE_CONFIRM_FRAMES = 3
MOTORCYCLE_DEPARTURE_RELEASE_CONFIRM_FRAMES = 4
MOTORCYCLE_LANE_CHANGE_CONFIRM_FRAMES = 5
MOTORCYCLE_LANE_CHANGE_HOLD_FRAMES = 12


def normalize_vehicle_class(vehicle_class):
    """统一车型名称，避免不同函数中 class / class_name / vehicle_class 字段不一致。"""
    return str(vehicle_class or "").strip().lower()


def is_motorcycle_class(vehicle_class):
    return normalize_vehicle_class(vehicle_class) in {"motorcycle", "motorbike", "moto"}


def get_vehicle_param(vehicle_class, name, default=None):
    """按车型返回参数。普通汽车/货车/客车使用全局参数，摩托车使用 MOTORCYCLE_* 参数。"""
    if is_motorcycle_class(vehicle_class):
        moto_name = "MOTORCYCLE_" + name
        if moto_name in globals():
            return globals()[moto_name]
    if name in globals():
        return globals()[name]
    return default

def get_min_conf_by_class(vehicle_class):
    """
    按车型设置检测结果的二次置信度阈值。

    YOLO_CONF 是模型入口阈值，需要设低一点，保证远处小目标摩托车先进入候选；
    本函数再对 car / truck / bus / motorcycle 分别过滤，避免普通车辆误检增加。
    """
    vehicle_class = normalize_vehicle_class(vehicle_class)

    if vehicle_class == "motorcycle":
        return MOTORCYCLE_MIN_CONF

    if vehicle_class == "car":
        return CAR_MIN_CONF

    if vehicle_class == "truck":
        return TRUCK_MIN_CONF

    if vehicle_class == "bus":
        return BUS_MIN_CONF

    return 1.0


@contextmanager
def vehicle_type_parameter_context(vehicle_class):
    """在单车测距期间临时切换为摩托车参数，结束后恢复普通车辆参数。"""
    if not is_motorcycle_class(vehicle_class):
        yield
        return

    mapped_names = [
        "RIGHT_EDGE_ROW_MIN_PIXELS",
        "RIGHT_EDGE_MIN_VALID_ROWS",
        "RIGHT_EDGE_BAND_MIN",
        "RIGHT_EDGE_BAND_MAX",
        "LANE_ASSIGN_SCAN_TOP_RATIO",
        "LANE_ASSIGN_SCAN_BOTTOM_RATIO",
        "LANE_ASSIGN_WIDTH_KEEP_RATIO",
        "LANE_ASSIGN_Y_PERCENTILE",
        "LANE_ASSIGN_CONFIRM_FRAMES",
        "RIGHT_EDGE_FOOTPRINT_TOP_RATIO",
        "RIGHT_EDGE_FOOTPRINT_BOTTOM_RATIO",
        "RIGHT_EDGE_FOOTPRINT_WIDTH_KEEP_RATIO",
        "RIGHT_EDGE_FOOTPRINT_Y_PERCENTILE",
        "RIGHT_EDGE_RIGHT_X_PERCENTILE",
        "RIGHT_EDGE_LOCAL_Y_WINDOW_RATIO",
        "EDGE_SANITY_MIN_ABS_JUMP_PX",
        "EDGE_SANITY_MAX_ABS_JUMP_PX",
        "EDGE_SANITY_BBOX_WIDTH_RATIO",
        "MEASUREMENT_BOTTOM_CLIP_MARGIN_PX",
        "MEASUREMENT_SIDE_CLIP_MARGIN_PX",
        "MEASUREMENT_MAX_BBOX_AREA_RATIO",
        "MEASUREMENT_MAX_BBOX_HEIGHT_RATIO",
        "MEASUREMENT_MAX_BBOX_WIDTH_RATIO",
        "MEASUREMENT_MAX_BBOX_WIDTH_LANE_RATIO",
        "MEASUREMENT_LOW_REF_MARGIN_PX",
        "DEPARTURE_CONFIRM_FRAMES",
        "DEPARTURE_RELEASE_CONFIRM_FRAMES",
        "LANE_CHANGE_CONFIRM_FRAMES",
        "LANE_CHANGE_HOLD_FRAMES",
    ]

    old_values = {name: globals()[name] for name in mapped_names if name in globals()}

    for name in mapped_names:
        moto_name = "MOTORCYCLE_" + name
        if moto_name in globals() and name in globals():
            globals()[name] = globals()[moto_name]

    try:
        yield
    finally:
        globals().update(old_values)

# 右换道确认帧数略降低：车辆中心已经连续进入相邻右车道时，及时切换测距基准线，
# 避免旧车道右边界继续参与测距而产生过大的负距离。

# =========================
# 颜色（OpenCV: BGR）
# =========================
CONTINUOUS_COLOR = (255, 255, 0)       # 亮青色：稳定车道边界
PAINT_COLOR = (255, 0, 255)            # 洋红色：实际标线段
GHOST_COLOR = (160, 160, 0)            # 暗青色：遮挡区域参考线

VEHICLE_BOX_COLOR = (0, 255, 0)        # 普通车辆：绿色
MOTORCYCLE_BOX_COLOR = (0, 0, 255)     # 摩托车：红色
VEHICLE_EDGE_COLOR = (255, 255, 255)

DIST_SAFE_COLOR = (0, 220, 0)
DIST_NEAR_COLOR = (0, 220, 255)
DIST_TOUCH_COLOR = (0, 128, 255)
DIST_CROSS_COLOR = (0, 0, 255)


# =========================================================
# 基础工具
# =========================================================
def load_lane_map(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到稳定车道地图：{path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_roi_mask(shape, roi_polygon):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(roi_polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


def sample_polyline(coeff, y_min, y_max, roi_mask, step=3):
    h, w = roi_mask.shape[:2]
    pts = []

    for y in range(int(y_min), int(y_max) + 1, step):
        x = np.polyval(coeff, y)
        x_i = int(round(x))
        y_i = int(round(y))

        if 0 <= x_i < w and 0 <= y_i < h and roi_mask[y_i, x_i] > 0:
            pts.append([x_i, y_i])

    return pts


def ensure_odd(v):
    v = int(v)
    return v if v % 2 == 1 else v + 1


def blend_color_by_soft_mask(base, color_bgr, soft_mask, alpha):
    if alpha <= 0:
        return base

    mask_f = soft_mask.astype(np.float32) / 255.0
    if np.max(mask_f) < 1e-6:
        return base

    a = (mask_f * alpha)[..., None]
    base_f = base.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)

    out = base_f * (1.0 - a) + color * a
    return np.clip(out, 0, 255).astype(np.uint8)


def status_to_color(status):
    if status == "SAFE":
        return DIST_SAFE_COLOR
    if status == "NEAR":
        return DIST_NEAR_COLOR
    if status == "TOUCH":
        return DIST_TOUCH_COLOR
    if status == "CROSS":
        return DIST_CROSS_COLOR
    return (200, 200, 200)


# =========================================================
# 车道线 mask 构建
# =========================================================
def build_lane_layers(frame_shape, lane_map):
    h, w = frame_shape[:2]

    roi_mask, roi_pts = make_roi_mask(frame_shape, lane_map["roi_polygon"])

    y_top = lane_map["y_top"]
    y_bottom = lane_map["y_bottom"]

    continuous_mask = np.zeros((h, w), dtype=np.uint8)
    paint_mask = np.zeros((h, w), dtype=np.uint8)
    ghost_mask = np.zeros((h, w), dtype=np.uint8)

    for lane in lane_map["stable_lanes"]:
        coeff = np.array(lane["coeff"], dtype=np.float64)

        full_pts = sample_polyline(coeff, y_top, y_bottom, roi_mask, step=3)

        if len(full_pts) >= 2:
            pts_np = np.array(full_pts, dtype=np.int32)

            cv2.polylines(
                continuous_mask,
                [pts_np],
                False,
                255,
                CONTINUOUS_THICKNESS,
                cv2.LINE_AA,
            )

            cv2.polylines(
                ghost_mask,
                [pts_np],
                False,
                255,
                GHOST_THICKNESS,
                cv2.LINE_AA,
            )

        for seg in lane.get("paint_segments_y", []):
            seg_y1, seg_y2 = int(seg[0]), int(seg[1])
            seg_pts = sample_polyline(coeff, seg_y1, seg_y2, roi_mask, step=3)

            if len(seg_pts) >= 2:
                seg_pts_np = np.array(seg_pts, dtype=np.int32)

                cv2.polylines(
                    paint_mask,
                    [seg_pts_np],
                    False,
                    255,
                    PAINT_THICKNESS,
                    cv2.LINE_AA,
                )

    return {
        "roi_mask": roi_mask,
        "roi_pts": roi_pts,
        "continuous_mask": continuous_mask,
        "paint_mask": paint_mask,
        "ghost_mask": ghost_mask,
    }


# =========================================================
# 自定义车辆跟踪器
# =========================================================
class ConservativeVehicleTracker:
    """
    单车单 ID 稳定跟踪器：
    1. 先用检测结果匹配已有 track；
    2. 未匹配检测在新建 ID 前，先检查是否与已有 track 属于同一辆车；
    3. 同一帧重复检测只保留一个显示 ID；
    4. track 合并时保留较早的 display_id，避免同车出现两个编号；
    5. 适当延长 missing 保留时间，减少短时漏检后重新分配 ID。
    """

    def __init__(self):
        self.tracks = {}
        self.next_track_id = 1
        self.next_display_id = 1

    @staticmethod
    def box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter
        if union <= 0:
            return 0.0

        return inter / union

    @staticmethod
    def box_intersection_over_min(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))

        return inter / max(1, min(area_a, area_b))

    @staticmethod
    def box_center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    @staticmethod
    def box_diag(box):
        x1, y1, x2, y2 = box
        return float(np.hypot(max(1, x2 - x1), max(1, y2 - y1)))

    @staticmethod
    def box_area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def area_ratio_ok(area_a, area_b):
        if area_a <= 0 or area_b <= 0:
            return False
        ratio = area_a / max(1, area_b)
        return TRACK_DUP_AREA_RATIO_MIN <= ratio <= TRACK_DUP_AREA_RATIO_MAX

    def boxes_likely_same_vehicle(
        self,
        box_a,
        box_b,
        iou_th=None,
        center_ratio_th=None,
        containment_th=None,
    ):
        if iou_th is None:
            iou_th = TRACK_DUP_IOU_TH
        if center_ratio_th is None:
            center_ratio_th = TRACK_DUP_CENTER_RATIO
        if containment_th is None:
            containment_th = TRACK_DUP_CONTAINMENT_TH

        iou = self.box_iou(box_a, box_b)
        if iou >= iou_th:
            return True

        contain = self.box_intersection_over_min(box_a, box_b)
        if contain >= containment_th:
            return True

        area_a = self.box_area(box_a)
        area_b = self.box_area(box_b)
        if not self.area_ratio_ok(area_a, area_b):
            return False

        center_a = self.box_center(box_a)
        center_b = self.box_center(box_b)
        center_dist = float(np.linalg.norm(center_a - center_b))

        scale = max(self.box_diag(box_a), self.box_diag(box_b), 1.0)
        center_ratio = center_dist / scale

        return center_ratio <= center_ratio_th

    def match_detection_to_track(self, det, used_track_ids):
        best_id = None
        best_score = -1.0

        det_box = det["box"]
        det_center = self.box_center(det_box)

        for tid, trk in self.tracks.items():
            if tid in used_track_ids:
                continue

            trk_box = trk["box"]
            trk_center = self.box_center(trk_box)

            iou = self.box_iou(det_box, trk_box)
            contain = self.box_intersection_over_min(det_box, trk_box)

            center_dist = float(np.linalg.norm(det_center - trk_center))
            scale = max(self.box_diag(det_box), self.box_diag(trk_box), 1.0)
            center_ratio = center_dist / scale

            # 中心距离过大且无重叠时，不匹配，避免串 ID。
            if (
                iou < CUSTOM_MATCH_IOU_TH
                and contain < NEW_TRACK_DUP_CONTAINMENT_TH
                and center_ratio > CUSTOM_MATCH_CENTER_RATIO
            ):
                continue

            center_score = max(0.0, 1.0 - center_ratio)
            max_missing = int(
                get_vehicle_param(
                    trk.get("class", ""),
                    "CUSTOM_TRACK_MAX_MISSING",
                    CUSTOM_TRACK_MAX_MISSING,
                )
            )
            recent_score = max(0, max_missing - trk["missing"]) / max(1, max_missing)

            # contain 可处理“同一辆车一个大框一个小框”的重复检测。
            score = 0.50 * iou + 0.25 * center_score + 0.15 * contain + 0.10 * recent_score

            if score > best_score:
                best_score = score
                best_id = tid

        return best_id

    def find_duplicate_track_for_detection(self, det):
        """未匹配检测新建 ID 前，先找是否已经有同车 track。"""
        best_id = None
        best_score = -1.0
        det_box = det["box"]

        for tid, trk in self.tracks.items():
            trk_box = trk["box"]
            if not self.boxes_likely_same_vehicle(
                det_box,
                trk_box,
                iou_th=NEW_TRACK_DUP_IOU_TH,
                center_ratio_th=NEW_TRACK_DUP_CENTER_RATIO,
                containment_th=NEW_TRACK_DUP_CONTAINMENT_TH,
            ):
                continue

            iou = self.box_iou(det_box, trk_box)
            contain = self.box_intersection_over_min(det_box, trk_box)
            center_ratio = float(
                np.linalg.norm(self.box_center(det_box) - self.box_center(trk_box))
            ) / max(self.box_diag(det_box), self.box_diag(trk_box), 1.0)

            score = 0.45 * iou + 0.35 * contain + 0.20 * max(0.0, 1.0 - center_ratio)
            if score > best_score:
                best_score = score
                best_id = tid

        return best_id

    def update_track_from_detection(self, trk, det):
        old_box = np.array(trk["box"], dtype=np.float32)
        new_box = np.array(det["box"], dtype=np.float32)

        smooth_box = (
            CUSTOM_BOX_SMOOTH_ALPHA * new_box
            + (1.0 - CUSTOM_BOX_SMOOTH_ALPHA) * old_box
        )

        trk["box"] = smooth_box.tolist()
        trk["mask"] = det["mask"].copy()
        trk["class"] = det["class"]
        trk["conf"] = det["conf"]
        trk["missing"] = 0
        trk["hits"] += 1

    def merge_duplicate_tracks(self):
        changed = True

        while changed:
            changed = False
            ids = list(self.tracks.keys())
            remove_ids = set()

            for i in range(len(ids)):
                id_a = ids[i]
                if id_a in remove_ids or id_a not in self.tracks:
                    continue

                for j in range(i + 1, len(ids)):
                    id_b = ids[j]
                    if id_b in remove_ids or id_b not in self.tracks:
                        continue

                    a = self.tracks[id_a]
                    b = self.tracks[id_b]

                    if not self.boxes_likely_same_vehicle(a["box"], b["box"]):
                        continue

                    max_missing_a = int(
                        get_vehicle_param(a.get("class", ""), "CUSTOM_TRACK_MAX_MISSING", CUSTOM_TRACK_MAX_MISSING)
                    )
                    max_missing_b = int(
                        get_vehicle_param(b.get("class", ""), "CUSTOM_TRACK_MAX_MISSING", CUSTOM_TRACK_MAX_MISSING)
                    )

                    score_a = (
                        2.0 * (max_missing_a - a["missing"])
                        + 0.5 * a["hits"]
                        + 1.0 * a["conf"]
                        + 0.00001 * self.box_area(a["box"])
                    )

                    score_b = (
                        2.0 * (max_missing_b - b["missing"])
                        + 0.5 * b["hits"]
                        + 1.0 * b["conf"]
                        + 0.00001 * self.box_area(b["box"])
                    )

                    keep_id, drop_id = (id_a, id_b) if score_a >= score_b else (id_b, id_a)

                    # 保留更早出现的显示编号，避免同车编号跳变。
                    self.tracks[keep_id]["display_id"] = min(
                        self.tracks[keep_id]["display_id"],
                        self.tracks[drop_id]["display_id"],
                    )
                    self.tracks[keep_id]["hits"] = max(
                        self.tracks[keep_id]["hits"],
                        self.tracks[drop_id]["hits"],
                    )
                    self.tracks[keep_id]["missing"] = min(
                        self.tracks[keep_id]["missing"],
                        self.tracks[drop_id]["missing"],
                    )

                    remove_ids.add(drop_id)
                    changed = True
                    break

                if changed:
                    break

            for tid in remove_ids:
                if tid in self.tracks:
                    del self.tracks[tid]

    def dedupe_final_boxes(self, final_boxes):
        """最终输出前再做一次同帧去重，保证画面上同一辆车只显示一个 ID。"""
        kept = []

        for item in sorted(
            final_boxes,
            key=lambda x: (x.get("display_id", 10**9), -float(x.get("conf", 0.0))),
        ):
            duplicate = False
            for old in kept:
                if self.boxes_likely_same_vehicle(
                    item["box"],
                    old["box"],
                    iou_th=FINAL_BOX_DUP_IOU_TH,
                    center_ratio_th=FINAL_BOX_DUP_CENTER_RATIO,
                    containment_th=FINAL_BOX_DUP_CONTAINMENT_TH,
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(item)

        return kept

    def update(self, detections, frame_shape, roi_mask):
        h, w = frame_shape[:2]

        used_track_ids = set()
        used_det_indices = set()
        updated_track_ids = set()

        detections = sorted(detections, key=lambda d: d["conf"], reverse=True)

        # 1. 当前检测匹配已有 track。
        for det_idx, det in enumerate(detections):
            matched_id = self.match_detection_to_track(det, used_track_ids)

            if matched_id is None:
                continue

            trk = self.tracks[matched_id]
            self.update_track_from_detection(trk, det)

            used_track_ids.add(matched_id)
            used_det_indices.add(det_idx)
            updated_track_ids.add(matched_id)

        # 2. 未匹配检测：新建 track 前先查重。
        #    如果同一帧同一辆车被 YOLO 分成两个框，第二个框不会再生成新 ID。
        for det_idx, det in enumerate(detections):
            if det_idx in used_det_indices:
                continue

            duplicate_track_id = self.find_duplicate_track_for_detection(det)
            if duplicate_track_id is not None:
                # 如果该 track 本帧还没更新，则用这个检测更新；否则直接丢弃该重复检测。
                if duplicate_track_id not in updated_track_ids:
                    self.update_track_from_detection(self.tracks[duplicate_track_id], det)
                    updated_track_ids.add(duplicate_track_id)
                used_det_indices.add(det_idx)
                continue

            tid = self.next_track_id
            self.next_track_id += 1

            self.tracks[tid] = {
                "box": list(map(float, det["box"])),
                "mask": det["mask"].copy(),
                "class": det["class"],
                "conf": det["conf"],
                "missing": 0,
                "hits": 1,
                "display_id": self.next_display_id,
            }

            self.next_display_id += 1
            updated_track_ids.add(tid)

        # 3. 未更新 track，missing + 1。
        remove_ids = []

        for tid, trk in self.tracks.items():
            if tid not in updated_track_ids:
                trk["missing"] += 1

                max_missing = int(
                    get_vehicle_param(
                        trk.get("class", ""),
                        "CUSTOM_TRACK_MAX_MISSING",
                        CUSTOM_TRACK_MAX_MISSING,
                    )
                )
                if trk["missing"] > max_missing:
                    remove_ids.append(tid)

        for tid in remove_ids:
            if tid in self.tracks:
                del self.tracks[tid]

        # 4. 合并重复 track。
        self.merge_duplicate_tracks()

        # 5. 输出 final mask / box。
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_boxes = []
        final_contours = []

        for tid, trk in self.tracks.items():
            min_hits = int(
                get_vehicle_param(
                    trk.get("class", ""),
                    "CUSTOM_TRACK_MIN_HITS",
                    CUSTOM_TRACK_MIN_HITS,
                )
            )
            if trk["hits"] < min_hits:
                continue

            if trk["missing"] <= 1:
                final_mask = cv2.bitwise_or(final_mask, trk["mask"])

            draw_box = trk["missing"] == 0

            if draw_box:
                x1, y1, x2, y2 = np.array(trk["box"]).astype(int)
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))

                final_boxes.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "mask": trk["mask"].copy(),
                        "class": trk["class"],
                        "conf": float(trk["conf"]),
                        "display_id": int(trk["display_id"]),
                        "missing": int(trk["missing"]),
                    }
                )

        final_boxes = self.dedupe_final_boxes(final_boxes)

        final_mask = cv2.bitwise_and(final_mask, roi_mask)

        if SHOW_VEHICLE_MASK_EDGE:
            contours, _ = cv2.findContours(
                final_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            final_contours = contours

        return final_mask, final_boxes, final_contours


VEHICLE_TRACKER = ConservativeVehicleTracker()


# =========================================================
# 车辆检测辅助函数
# =========================================================
def bbox_from_mask(mask, fallback_box, pad=2, vehicle_class=""):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return fallback_box

    h, w = mask.shape[:2]

    mx1 = max(0, int(xs.min()) - pad)
    my1 = max(0, int(ys.min()) - pad)
    mx2 = min(w - 1, int(xs.max()) + pad)
    my2 = min(h - 1, int(ys.max()) + pad)

    if mx2 <= mx1 or my2 <= my1:
        return fallback_box

    fx1, fy1, fx2, fy2 = fallback_box

    mask_area = max(1, (mx2 - mx1) * (my2 - my1))
    fallback_area = max(1, (fx2 - fx1) * (fy2 - fy1))

    mask_center = np.array([(mx1 + mx2) * 0.5, (my1 + my2) * 0.5])
    fallback_center = np.array([(fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5])

    fallback_diag = max(1.0, float(np.hypot(fx2 - fx1, fy2 - fy1)))
    center_shift = float(np.linalg.norm(mask_center - fallback_center)) / fallback_diag

    if mask_area > fallback_area * float(get_vehicle_param(vehicle_class, "MASK_BOX_MAX_AREA_RATIO", MASK_BOX_MAX_AREA_RATIO)):
        return fallback_box

    if mask_area < fallback_area * float(get_vehicle_param(vehicle_class, "MASK_BOX_MIN_AREA_RATIO", MASK_BOX_MIN_AREA_RATIO)):
        return fallback_box

    if center_shift > float(get_vehicle_param(vehicle_class, "MASK_BOX_MAX_CENTER_SHIFT_RATIO", MASK_BOX_MAX_CENTER_SHIFT_RATIO)):
        return fallback_box

    return [mx1, my1, mx2, my2]


def suppress_duplicate_detections(detections):
    """同一帧 YOLO 检测去重，避免一辆车同时生成两个检测框。"""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []

    tracker_ref = ConservativeVehicleTracker()

    for det in detections:
        duplicate = False

        for old in kept:
            # 不按类别强制区分，避免 car/truck/motorcycle 类别抖动导致双框。
            iou = ConservativeVehicleTracker.box_iou(det["box"], old["box"])
            contain = ConservativeVehicleTracker.box_intersection_over_min(det["box"], old["box"])

            center_dist = float(
                np.linalg.norm(
                    ConservativeVehicleTracker.box_center(det["box"])
                    - ConservativeVehicleTracker.box_center(old["box"])
                )
            )
            scale = max(
                ConservativeVehicleTracker.box_diag(det["box"]),
                ConservativeVehicleTracker.box_diag(old["box"]),
                1.0,
            )
            center_ratio = center_dist / scale

            same_by_center = (
                tracker_ref.area_ratio_ok(
                    ConservativeVehicleTracker.box_area(det["box"]),
                    ConservativeVehicleTracker.box_area(old["box"]),
                )
                and center_ratio <= NMS_CENTER_RATIO
            )

            if iou >= NMS_IOU_TH or contain >= NMS_CONTAINMENT_TH or same_by_center:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


# =========================================================
# 车辆检测
# =========================================================
def detect_vehicle_mask_and_boxes(model, frame, roi_mask):
    """
    车辆检测 + mask 提取 + 自定义跟踪入口。

    本版重点解决远处摩托车漏检问题：
    1. YOLO_CONF 作为模型入口阈值，设置较低，让 motorcycle 小目标先进入候选；
    2. 对 car / truck / bus / motorcycle 再做分车型二次置信度过滤；
    3. motorcycle 使用更低置信度和更低 ROI 交集阈值；
    4. 普通车辆仍保持较高置信度，避免汽车误检明显增加。
    """
    h, w = frame.shape[:2]
    detections = []

    results = model.predict(
        frame,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )

    if not results:
        return VEHICLE_TRACKER.update([], frame.shape, roi_mask)

    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return VEHICLE_TRACKER.update([], frame.shape, roi_mask)

    names = r.names

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    has_seg = r.masks is not None and r.masks.xy is not None

    for i, (box, cls_id, conf) in enumerate(zip(boxes_xyxy, cls_ids, confs)):
        raw_cls_name = names.get(int(cls_id), str(cls_id))
        cls_name = normalize_vehicle_class(raw_cls_name)

        # 只保留车辆类别。注意：person 不进入正式测距，避免行人/路边物体误检。
        if cls_name not in VEHICLE_CLASS_NAMES:
            continue

        # 分车型二次置信度过滤：摩托车低阈值，汽车/货车/客车高阈值。
        min_conf = get_min_conf_by_class(cls_name)
        if float(conf) < float(min_conf):
            continue

        # 摩托车体积小，远处与 ROI 的交集很少，因此使用更低的 ROI 交集阈值。
        min_roi_intersection = int(
            get_vehicle_param(cls_name, "MIN_ROI_INTERSECTION", MIN_ROI_INTERSECTION)
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        roi_crop = roi_mask[y1:y2, x1:x2]
        if np.count_nonzero(roi_crop) < min_roi_intersection:
            continue

        # 保留完整 YOLO 分割 mask 用于车辆边缘测距。
        # ROI 只用于过滤目标和最终遮挡显示，不能提前裁掉车辆 mask，
        # 否则车辆贴近 ROI 边界时，测距边缘会被 ROI 截断。
        det_mask_full = np.zeros((h, w), dtype=np.uint8)

        if has_seg and i < len(r.masks.xy):
            polygon = r.masks.xy[i]

            if polygon is not None and len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(det_mask_full, [pts], 255)
            else:
                cv2.rectangle(det_mask_full, (x1, y1), (x2, y2), 255, -1)
        else:
            cv2.rectangle(det_mask_full, (x1, y1), (x2, y2), 255, -1)

        det_mask_roi = cv2.bitwise_and(det_mask_full, roi_mask)

        if np.count_nonzero(det_mask_roi) < min_roi_intersection:
            continue

        fallback_box = [int(x1), int(y1), int(x2), int(y2)]
        mask_box = bbox_from_mask(
            det_mask_full,
            fallback_box,
            pad=3,
            vehicle_class=cls_name,
        )

        detections.append(
            {
                "box": mask_box,
                "mask": det_mask_full,
                "class": cls_name,
                "conf": float(conf),
            }
        )

    detections = suppress_duplicate_detections(detections)

    return VEHICLE_TRACKER.update(detections, frame.shape, roi_mask)

def make_soft_vehicle_mask(vehicle_mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (VEHICLE_MASK_DILATE_X, VEHICLE_MASK_DILATE_Y),
    )
    hard_mask = cv2.dilate(vehicle_mask, kernel, iterations=1)

    blur_k = ensure_odd(SOFT_MASK_BLUR_KSIZE)
    soft_mask = cv2.GaussianBlur(hard_mask, (blur_k, blur_k), 0)

    return hard_mask, soft_mask


# =========================================================
# 距离计算
# =========================================================
def lane_x_at_y(lane, y):
    coeff = np.array(lane["coeff"], dtype=np.float64)
    return float(np.polyval(coeff, float(y)))


def get_sorted_lane_positions(lane_map, y, frame_width):
    positions = []

    for idx, lane in enumerate(lane_map["stable_lanes"]):
        x = lane_x_at_y(lane, y)

        if np.isfinite(x) and -frame_width <= x <= frame_width * 2:
            positions.append(
                {
                    "lane_id": idx,
                    "x": float(x),
                    "lane": lane,
                }
            )

    positions.sort(key=lambda item: item["x"])
    return positions



def collect_mask_row_profile(vehicle_mask, box, top_ratio, bottom_ratio):
    """
    逐行提取车辆 mask 的左/右边缘轮廓。

    返回的每一行包含：y、left_x、right_x、center_x、row_width。
    这里不直接用 bbox，因为 bbox 受车身阴影、分割框平滑和类别框影响较大；
    对于车道归属，应优先使用靠近路面的可见车辆轮廓中心。
    """
    x1, y1, x2, y2 = map(int, box)
    h = max(1, y2 - y1)

    y_low = max(y1, int(round(y1 + top_ratio * h)))
    y_high = min(y2, int(round(y1 + bottom_ratio * h)))

    if vehicle_mask is None or y_high <= y_low:
        return [], int(y_low), int(y_high)

    crop = vehicle_mask[y_low:y_high + 1, x1:x2 + 1]
    if crop.size == 0:
        return [], int(y_low), int(y_high)

    rows = []
    for row_idx in range(crop.shape[0]):
        xs = np.flatnonzero(crop[row_idx] > 0)
        if len(xs) < RIGHT_EDGE_ROW_MIN_PIXELS:
            continue

        left = int(xs.min())
        right = int(xs.max())
        width = right - left + 1
        if width < RIGHT_EDGE_ROW_MIN_PIXELS:
            continue

        y = int(y_low + row_idx)
        left_x = float(x1 + left)
        right_x = float(x1 + right)

        rows.append(
            {
                "y": y,
                "left_x": left_x,
                "right_x": right_x,
                "center_x": 0.5 * (left_x + right_x),
                "row_width": int(width),
            }
        )

    return rows, int(y_low), int(y_high)


def filter_stable_width_rows(rows, keep_ratio):
    """
    排除车头/车尾端部的窄行，只保留车辆主体轮廓行。
    这种过滤比固定用 bbox 宽度更稳，因为远处小车和大货车尺度差异很大。
    """
    if not rows:
        return []

    max_width = max(int(r["row_width"]) for r in rows)
    min_width = max(RIGHT_EDGE_ROW_MIN_PIXELS, int(round(max_width * keep_ratio)))
    return [r for r in rows if int(r["row_width"]) >= min_width]


def pick_rows_near_target_y(rows, target_y, window_px):
    if not rows:
        return []

    near = [r for r in rows if abs(float(r["y"]) - float(target_y)) <= float(window_px)]
    if len(near) >= RIGHT_EDGE_MIN_VALID_ROWS:
        return near

    # 近邻行不足时，取距离 target_y 最近的若干行，避免直接退回 bbox。
    rows_sorted = sorted(rows, key=lambda r: abs(float(r["y"]) - float(target_y)))
    return rows_sorted[: max(RIGHT_EDGE_MIN_VALID_ROWS, min(7, len(rows_sorted)))]


def get_vehicle_road_profile(vehicle_mask, box):
    """
    v4.3 车道归属点：车辆底部可见轮廓中心，而不是 bbox 几何中心。

    bbox 中心在高空斜视画面中经常偏向车顶/车尾，导致车辆还在原车道时
    就被错误归到相邻车道。这里使用下半部稳定宽度行的左右边界中点，
    更接近车辆在路面上的横向占位中心。
    """
    x1, y1, x2, y2 = map(int, box)
    h = max(1, y2 - y1)
    fallback_y = get_vehicle_ref_y(box)

    fallback = {
        "center_x": 0.5 * (x1 + x2),
        "left_x": float(x1),
        "right_x": float(x2),
        "y": int(fallback_y),
        "source": "bbox_center_fallback",
        "valid_rows": 0,
        "row_width": int(max(1, x2 - x1)),
        "y_low": int(fallback_y),
        "y_high": int(fallback_y),
    }

    rows, y_low, y_high = collect_mask_row_profile(
        vehicle_mask,
        box,
        LANE_ASSIGN_SCAN_TOP_RATIO,
        LANE_ASSIGN_SCAN_BOTTOM_RATIO,
    )
    rows = filter_stable_width_rows(rows, LANE_ASSIGN_WIDTH_KEEP_RATIO)

    if len(rows) < RIGHT_EDGE_MIN_VALID_ROWS:
        fallback["y_low"] = y_low
        fallback["y_high"] = y_high
        return fallback

    ys = np.array([r["y"] for r in rows], dtype=np.float64)
    target_y = float(np.percentile(ys, LANE_ASSIGN_Y_PERCENTILE))
    window_px = max(4, int(round(h * 0.06)))
    local_rows = pick_rows_near_target_y(rows, target_y, window_px)

    left_x = float(np.median([r["left_x"] for r in local_rows]))
    right_x = float(np.median([r["right_x"] for r in local_rows]))
    center_x = 0.5 * (left_x + right_x)
    y = int(round(np.median([r["y"] for r in local_rows])))
    row_width = int(round(np.median([r["row_width"] for r in local_rows])))

    return {
        "center_x": float(center_x),
        "left_x": float(left_x),
        "right_x": float(right_x),
        "y": int(y),
        "source": "mask_road_profile",
        "valid_rows": int(len(rows)),
        "row_width": int(row_width),
        "y_low": int(y_low),
        "y_high": int(y_high),
    }

def get_vehicle_ref_y(box):
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)
    ref_y = int(round(y2 - REF_Y_FROM_BOTTOM_RATIO * h))
    return max(y1, min(y2, ref_y))


def get_vehicle_right_edge_at_y(vehicle_mask, box, ref_y):
    """
    在车辆底部附近横截面上提取车辆右侧边界。

    v4 原版直接取 band 内所有 mask 像素的 xs.max()，容易被分割毛刺、
    后视镜、车身小突出部拉到过右的位置。这里改为：
    1. 在 ref_y 附近取一个横向测距带；
    2. 每一行单独找该行 mask 的最右像素；
    3. 对多行最右像素取高分位数，而不是单点最大值；
    4. 有效行不足时退回 bbox 右边界。
    """
    x1, y1, x2, y2 = map(int, box)
    h = max(1, y2 - y1)

    band = int(round(h * RIGHT_EDGE_BAND_RATIO))
    band = max(RIGHT_EDGE_BAND_MIN, min(RIGHT_EDGE_BAND_MAX, band))

    y_low = max(y1, int(ref_y) - band)
    y_high = min(y2, int(ref_y) + band)

    fallback = {
        "x": float(x2),
        "source": "bbox_fallback",
        "valid_rows": 0,
        "band": int(band),
        "y_low": int(y_low),
        "y_high": int(y_high),
    }

    if vehicle_mask is None:
        return fallback

    crop = vehicle_mask[y_low:y_high + 1, x1:x2 + 1]
    if crop.size == 0:
        return fallback

    row_rights = []

    for row_idx in range(crop.shape[0]):
        xs = np.flatnonzero(crop[row_idx] > 0)

        # 每行至少要有几个 mask 像素，避免单个噪点控制右边界
        if len(xs) < RIGHT_EDGE_ROW_MIN_PIXELS:
            continue

        row_rights.append(float(x1 + xs.max()))

    if len(row_rights) < RIGHT_EDGE_MIN_VALID_ROWS:
        return fallback

    right_x = float(np.percentile(row_rights, RIGHT_EDGE_PERCENTILE))
    right_x = max(float(x1), min(float(x2), right_x))

    return {
        "x": right_x,
        "source": "mask_percentile",
        "valid_rows": int(len(row_rights)),
        "band": int(band),
        "y_low": int(y_low),
        "y_high": int(y_high),
    }



def get_vehicle_right_edge_near_lane(vehicle_mask, box, right_lane, frame_width):
    """
    v4.3 车辆右边缘提取：稳定下半部侧边轮廓法。

    v4.2 选“最接近右车道线”的低分位行，容易被车头/车尾窄边、后视镜、
    mask 毛刺拉到异常靠右的位置，导致 -1m、-3m 这类过激 CROSS。

    v4.3 改为：
    1. 扫描车辆下半部主体 mask 行；
    2. 用最大行宽的一定比例过滤窄端部；
    3. 选靠近车辆下半部的稳定 y 区域；
    4. 对局部右边缘 x 取 75% 分位，而不是单点最大或最小距离。
    """
    x1, y1, x2, y2 = map(int, box)
    h = max(1, y2 - y1)

    fallback_y = get_vehicle_ref_y(box)
    fallback_right_lane_x = lane_x_at_y(right_lane, fallback_y)
    fallback = {
        "x": float(x2),
        "y": int(fallback_y),
        "right_lane_x": float(fallback_right_lane_x),
        "distance_px": float(fallback_right_lane_x - x2),
        "source": "bbox_fallback",
        "valid_rows": 0,
        "band": 1,
        "y_low": int(fallback_y),
        "y_high": int(fallback_y),
        "row_width": 0,
    }

    rows, y_low, y_high = collect_mask_row_profile(
        vehicle_mask,
        box,
        RIGHT_EDGE_FOOTPRINT_TOP_RATIO,
        RIGHT_EDGE_FOOTPRINT_BOTTOM_RATIO,
    )
    rows = filter_stable_width_rows(rows, RIGHT_EDGE_FOOTPRINT_WIDTH_KEEP_RATIO)

    if len(rows) < RIGHT_EDGE_MIN_VALID_ROWS:
        fallback["y_low"] = y_low
        fallback["y_high"] = y_high
        fallback["band"] = max(0, y_high - y_low + 1)
        return fallback

    ys = np.array([r["y"] for r in rows], dtype=np.float64)
    target_y = float(np.percentile(ys, RIGHT_EDGE_FOOTPRINT_Y_PERCENTILE))
    window_px = max(4, int(round(h * RIGHT_EDGE_LOCAL_Y_WINDOW_RATIO)))
    local_rows = pick_rows_near_target_y(rows, target_y, window_px)

    right_x_values = np.array([r["right_x"] for r in local_rows], dtype=np.float64)
    target_right_x = float(np.percentile(right_x_values, RIGHT_EDGE_RIGHT_X_PERCENTILE))
    target_y_local = float(np.median([r["y"] for r in local_rows]))

    # 在局部候选中选择与分位右边界和目标 y 都接近的一行。
    # 这样画线端点仍落在真实 mask 轮廓上，而不是虚构的平均点。
    scale_x = max(1.0, float(np.std(right_x_values)) + 1.0)
    scale_y = max(1.0, float(window_px))

    best = min(
        local_rows,
        key=lambda r: (
            abs(float(r["right_x"]) - target_right_x) / scale_x
            + 0.35 * abs(float(r["y"]) - target_y_local) / scale_y
        ),
    )

    y = int(best["y"])
    vehicle_right_x = float(best["right_x"])
    right_lane_x = float(lane_x_at_y(right_lane, y))

    if not np.isfinite(right_lane_x):
        return fallback

    distance_px = float(right_lane_x - vehicle_right_x)

    return {
        "x": float(vehicle_right_x),
        "y": int(y),
        "right_lane_x": float(right_lane_x),
        "distance_px": float(distance_px),
        "source": "mask_footprint_side",
        "valid_rows": int(len(rows)),
        "band": int(y_high - y_low + 1),
        "y_low": int(y_low),
        "y_high": int(y_high),
        "row_width": int(best["row_width"]),
    }


def get_vehicle_side_edge_near_lane(vehicle_mask, box, boundary_lane, frame_width, side="right"):
    """
    根据车辆行驶方向提取测距侧边界。

    side="right"：取图像右侧车辆边缘，用于车辆从近往远行驶；
    side="left" ：取图像左侧车辆边缘，用于车辆从远往近行驶。
    """
    side = str(side).lower()
    if side not in {"left", "right"}:
        side = "right"

    x1, y1, x2, y2 = map(int, box)
    h = max(1, y2 - y1)

    fallback_y = get_vehicle_ref_y(box)
    fallback_lane_x = lane_x_at_y(boundary_lane, fallback_y)

    if side == "left":
        fallback_edge_x = float(x1)
        fallback_distance_px = float(x1 - fallback_lane_x)
        source_name = "bbox_fallback_left"
    else:
        fallback_edge_x = float(x2)
        fallback_distance_px = float(fallback_lane_x - x2)
        source_name = "bbox_fallback_right"

    fallback = {
        "x": float(fallback_edge_x),
        "y": int(fallback_y),
        "lane_x": float(fallback_lane_x),
        "distance_px": float(fallback_distance_px),
        "source": source_name,
        "valid_rows": 0,
        "band": 1,
        "y_low": int(fallback_y),
        "y_high": int(fallback_y),
        "row_width": 0,
    }

    rows, y_low, y_high = collect_mask_row_profile(
        vehicle_mask,
        box,
        RIGHT_EDGE_FOOTPRINT_TOP_RATIO,
        RIGHT_EDGE_FOOTPRINT_BOTTOM_RATIO,
    )
    rows = filter_stable_width_rows(rows, RIGHT_EDGE_FOOTPRINT_WIDTH_KEEP_RATIO)

    if len(rows) < RIGHT_EDGE_MIN_VALID_ROWS:
        fallback["y_low"] = y_low
        fallback["y_high"] = y_high
        fallback["band"] = max(0, y_high - y_low + 1)
        return fallback

    ys = np.array([r["y"] for r in rows], dtype=np.float64)
    target_y = float(np.percentile(ys, RIGHT_EDGE_FOOTPRINT_Y_PERCENTILE))
    window_px = max(4, int(round(h * RIGHT_EDGE_LOCAL_Y_WINDOW_RATIO)))
    local_rows = pick_rows_near_target_y(rows, target_y, window_px)

    if side == "left":
        edge_values = np.array([r["left_x"] for r in local_rows], dtype=np.float64)
        # 左侧边界取低分位，避免后视镜/毛刺把边缘拉得过左。
        target_edge_x = float(np.percentile(edge_values, 25))
        source_name = "mask_footprint_left_side"
    else:
        edge_values = np.array([r["right_x"] for r in local_rows], dtype=np.float64)
        target_edge_x = float(np.percentile(edge_values, RIGHT_EDGE_RIGHT_X_PERCENTILE))
        source_name = "mask_footprint_right_side"

    target_y_local = float(np.median([r["y"] for r in local_rows]))

    scale_x = max(1.0, float(np.std(edge_values)) + 1.0)
    scale_y = max(1.0, float(window_px))

    best = min(
        local_rows,
        key=lambda r: (
            abs(float(r["left_x" if side == "left" else "right_x"]) - target_edge_x) / scale_x
            + 0.35 * abs(float(r["y"]) - target_y_local) / scale_y
        ),
    )

    y = int(best["y"])
    vehicle_edge_x = float(best["left_x"] if side == "left" else best["right_x"])
    boundary_lane_x = float(lane_x_at_y(boundary_lane, y))

    if not np.isfinite(boundary_lane_x):
        return fallback

    if side == "left":
        distance_px = float(vehicle_edge_x - boundary_lane_x)
    else:
        distance_px = float(boundary_lane_x - vehicle_edge_x)

    return {
        "x": float(vehicle_edge_x),
        "y": int(y),
        "lane_x": float(boundary_lane_x),
        "distance_px": float(distance_px),
        "source": source_name,
        "valid_rows": int(len(rows)),
        "band": int(y_high - y_low + 1),
        "y_low": int(y_low),
        "y_high": int(y_high),
        "row_width": int(best["row_width"]),
    }


def classify_distance(distance_m):
    if not np.isfinite(distance_m):
        return "INVALID"

    if distance_m <= CROSS_DIST_M:
        return "CROSS"

    if distance_m <= TOUCH_DIST_M:
        return "TOUCH"

    if distance_m <= NEAR_DIST_M:
        return "NEAR"

    return "SAFE"


def assess_measurement_quality(box, frame_shape, lane_width_px, assignment_y):
    """
    v4.9：判断当前车辆是否适合做右侧距离计算。

    典型无效情况：
    1. 车辆贴近画面底边，已经半出画；
    2. 车辆 bbox 过大，透视畸变明显；
    3. bbox 横向跨度远大于局部车道宽度，说明当前框更像车辆纵向长度，
       直接用 mask 右缘容易把车尾/车身长边当成横向越线；
    4. 局部车道宽度异常，比例尺不可靠。
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame_shape[:2]
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    frame_area = max(1.0, float(w * h))
    bbox_area_ratio = (bw * bh) / frame_area
    bbox_height_ratio = bh / max(1.0, float(h))
    bbox_width_ratio = bw / max(1.0, float(w))
    lane_width_px = float(lane_width_px) if np.isfinite(float(lane_width_px)) else np.nan
    bbox_width_lane_ratio = bw / lane_width_px if np.isfinite(lane_width_px) and lane_width_px > 1 else np.inf

    reasons = []

    if not MEASUREMENT_QUALITY_GATE_ENABLE:
        return {
            "valid": True,
            "reason": "OK",
            "bbox_area_ratio": bbox_area_ratio,
            "bbox_height_ratio": bbox_height_ratio,
            "bbox_width_ratio": bbox_width_ratio,
            "bbox_width_lane_ratio": bbox_width_lane_ratio,
            "bottom_clipped": False,
            "side_clipped": False,
        }

    bottom_clipped = y2 >= h - MEASUREMENT_BOTTOM_CLIP_MARGIN_PX
    side_clipped = (
        x1 <= MEASUREMENT_SIDE_CLIP_MARGIN_PX
        or x2 >= w - MEASUREMENT_SIDE_CLIP_MARGIN_PX
    )
    ref_too_low = int(assignment_y) >= h - MEASUREMENT_LOW_REF_MARGIN_PX

    if bottom_clipped:
        reasons.append("bottom_clipped")
    if side_clipped:
        reasons.append("side_clipped")
    if ref_too_low:
        reasons.append("ref_y_too_low")
    if not np.isfinite(lane_width_px) or lane_width_px < MEASUREMENT_MIN_VALID_LANE_WIDTH_PX:
        reasons.append("bad_lane_width")
    if bbox_area_ratio > MEASUREMENT_MAX_BBOX_AREA_RATIO:
        reasons.append("bbox_area_too_large")
    if bbox_height_ratio > MEASUREMENT_MAX_BBOX_HEIGHT_RATIO:
        reasons.append("bbox_height_too_large")
    if bbox_width_ratio > MEASUREMENT_MAX_BBOX_WIDTH_RATIO:
        reasons.append("bbox_width_too_large")
    if bbox_width_lane_ratio > MEASUREMENT_MAX_BBOX_WIDTH_LANE_RATIO:
        reasons.append("bbox_wider_than_lane_unreliable")

    return {
        "valid": len(reasons) == 0,
        "reason": "OK" if not reasons else ";".join(reasons),
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_height_ratio": bbox_height_ratio,
        "bbox_width_ratio": bbox_width_ratio,
        "bbox_width_lane_ratio": bbox_width_lane_ratio,
        "bottom_clipped": bool(bottom_clipped),
        "side_clipped": bool(side_clipped),
    }


def lane_pair_key(left_lane, right_lane):
    return (int(left_lane["lane_id"]), int(right_lane["lane_id"]))


def locate_lane_pair_by_key(lane_positions, key):
    if key is None:
        return None, None

    for i in range(len(lane_positions) - 1):
        left = lane_positions[i]
        right = lane_positions[i + 1]

        if lane_pair_key(left, right) == tuple(key):
            return left, right

    return None, None


def locate_lane_pair_by_center(lane_positions, vehicle_center_x):
    """
    根据车辆中心点寻找所在车道，返回相邻两条车道线。
    """
    for i in range(len(lane_positions) - 1):
        lx = lane_positions[i]["x"]
        rx = lane_positions[i + 1]["x"]

        if lx <= vehicle_center_x <= rx:
            return lane_positions[i], lane_positions[i + 1]

    return None, None


class RightDistanceTemporalState:
    """
    距离计算的轻量级时序状态：
    1. 对车辆右侧测距点做指数平滑，降低分割抖动；
    2. 对车道归属做滞回，避免换道过程中右车道线突然跳到下一条；
    3. 根据 distance_m 的变化率和持续越线帧数给出初步换道状态。
    """

    def __init__(self):
        self.states = {}

    def _get_state(self, vehicle_id):
        vehicle_id = int(vehicle_id)
        if vehicle_id not in self.states:
            self.states[vehicle_id] = {
                "edge_x_smooth": None,
                "distance_m_smooth": None,
                "last_distance_m_smooth": None,
                "last_time_s": None,
                "accepted_lane_pair": None,
                "pending_lane_pair": None,
                "pending_count": 0,
                "cross_count": 0,
                "departure_count": 0,
                "release_count": 0,
                "lane_change_count": 0,
                "lane_change_hold": 0,
                "in_right_departure": False,
                "last_accepted_lane_pair_for_measurement": None,
                "last_edge_x_for_sanity": None,
                "last_bbox_center_x_for_sanity": None,
                "lane_pair_lock_count": 0,
                "lane_pair_last_switch_direction": "",
                "lane_revert_pending_count": 0,
                "last_frame_id": -1,
            }
        return self.states[vehicle_id]

    def select_lane_pair(self, vehicle_id, lane_positions, candidate_left, candidate_right, vehicle_center_x):
        """
        v4.9：带锁定的车道归属选择。

        v4.8 已经能在车辆进入右侧相邻车道时切换测距基准线，但在大车分割轮廓抖动时，
        车辆底部中心可能短时跳回旧车道，导致 lane 2-3 又回退到 lane 1-2，
        继而把右侧距离错误地计算成很大的负值。

        本函数在“向右完成一次车道对切换”后，短时间锁定当前车道对：
        - 不允许因为 1～2 帧中心抖动而立刻切回左侧旧车道；
        - 只有车辆中心明显回到当前车道左侧外部，并且连续满足若干帧，才允许左回退；
        - 正常继续向右切换仍然允许。
        """
        st = self._get_state(vehicle_id)

        if candidate_left is None or candidate_right is None:
            return None, None, None, None, 0, int(st.get("lane_pair_lock_count", 0))

        candidate_key = lane_pair_key(candidate_left, candidate_right)
        prev_key = st.get("accepted_lane_pair")

        if prev_key is None:
            st["accepted_lane_pair"] = candidate_key
            st["pending_lane_pair"] = None
            st["pending_count"] = 0
            st["lane_pair_lock_count"] = 0
            st["lane_pair_last_switch_direction"] = ""
            st["lane_revert_pending_count"] = 0
            return candidate_left, candidate_right, candidate_key, candidate_key, 0, 0

        # 锁定计数按帧递减。它只用于抑制“向右换道后的短时左回退”。
        if st.get("lane_pair_lock_count", 0) > 0:
            st["lane_pair_lock_count"] = max(0, int(st["lane_pair_lock_count"]) - 1)

        if candidate_key == prev_key:
            st["pending_lane_pair"] = None
            st["pending_count"] = 0
            st["lane_revert_pending_count"] = 0
            return candidate_left, candidate_right, prev_key, candidate_key, 0, int(st.get("lane_pair_lock_count", 0))

        prev_left, prev_right = locate_lane_pair_by_key(lane_positions, prev_key)
        if prev_left is None or prev_right is None:
            st["accepted_lane_pair"] = candidate_key
            st["pending_lane_pair"] = None
            st["pending_count"] = 0
            st["lane_pair_lock_count"] = 0
            st["lane_pair_last_switch_direction"] = ""
            st["lane_revert_pending_count"] = 0
            return candidate_left, candidate_right, candidate_key, candidate_key, 0, 0

        prev_left_x = float(prev_left["x"])
        prev_right_x = float(prev_right["x"])
        prev_width = max(1.0, prev_right_x - prev_left_x)
        center_ratio_prev = (float(vehicle_center_x) - prev_left_x) / prev_width

        # 只有车辆中心明显进入相邻车道后，才允许右车道线切换。
        # 这样车辆刚压线/越线时，仍然计算相对原右车道线的负距离。
        right_adjacent = candidate_key[0] == prev_key[1]
        left_adjacent = candidate_key[1] == prev_key[0]

        # v4.9：向右换道后，短时间不接受由于 mask 中心抖动造成的左回退。
        # 若车辆中心已经明显落到当前车道左外侧，并连续满足若干帧，才释放左回退。
        if (
            left_adjacent
            and st.get("lane_pair_lock_count", 0) > 0
            and st.get("lane_pair_last_switch_direction", "") == "RIGHT"
        ):
            if center_ratio_prev <= LANE_PAIR_LOCK_ALLOW_LEFT_REVERT_RATIO:
                st["lane_revert_pending_count"] = int(st.get("lane_revert_pending_count", 0)) + 1
            else:
                st["lane_revert_pending_count"] = 0

            if st["lane_revert_pending_count"] < LANE_PAIR_LOCK_LEFT_REVERT_CONFIRM_FRAMES:
                st["pending_lane_pair"] = None
                st["pending_count"] = 0
                prev_left, prev_right = locate_lane_pair_by_key(lane_positions, prev_key)
                return prev_left, prev_right, prev_key, candidate_key, 0, int(st.get("lane_pair_lock_count", 0))

            # 确认为真实左回退后释放锁定。
            st["lane_pair_lock_count"] = 0
            st["lane_pair_last_switch_direction"] = ""
            st["lane_revert_pending_count"] = 0

        if right_adjacent:
            allow_switch = center_ratio_prev >= 1.0 + LANE_ASSIGN_HYSTERESIS_RATIO
        elif left_adjacent:
            allow_switch = center_ratio_prev <= -LANE_ASSIGN_HYSTERESIS_RATIO
        else:
            allow_switch = True

        if not allow_switch:
            prev_left, prev_right = locate_lane_pair_by_key(lane_positions, prev_key)
            return prev_left, prev_right, prev_key, candidate_key, 0, int(st.get("lane_pair_lock_count", 0))

        if st["pending_lane_pair"] == candidate_key:
            st["pending_count"] += 1
        else:
            st["pending_lane_pair"] = candidate_key
            st["pending_count"] = 1

        if st["pending_count"] >= LANE_ASSIGN_CONFIRM_FRAMES:
            switch_direction = "RIGHT" if right_adjacent else ("LEFT" if left_adjacent else "JUMP")

            st["accepted_lane_pair"] = candidate_key
            st["pending_lane_pair"] = None
            st["pending_count"] = 0
            st["lane_revert_pending_count"] = 0

            if switch_direction == "RIGHT":
                st["lane_pair_lock_count"] = LANE_PAIR_LOCK_AFTER_RIGHT_CHANGE_FRAMES
                st["lane_pair_last_switch_direction"] = "RIGHT"
            else:
                st["lane_pair_lock_count"] = 0
                st["lane_pair_last_switch_direction"] = switch_direction

            return (
                candidate_left,
                candidate_right,
                candidate_key,
                candidate_key,
                0,
                int(st.get("lane_pair_lock_count", 0)),
            )

        prev_left, prev_right = locate_lane_pair_by_key(lane_positions, prev_key)
        return prev_left, prev_right, prev_key, candidate_key, int(st["pending_count"]), int(st.get("lane_pair_lock_count", 0))

    def reset_unreliable_measurement(self, vehicle_id):
        """
        v4.9：当目标被质量门控判为不可靠时，清空该 ID 的测距/换道时序状态。

        原因：近景大车半出画后，bbox 和 mask 的横向跨度可能远大于局部车道宽度，
        如果保留上一段平滑距离或车道归属，重新变为“可测”时会把旧的 CROSS /
        RIGHT_DEPARTURE 拖到新帧中，造成“没有换道却显示换道/偏离”的误判。
        """
        st = self._get_state(vehicle_id)

        st["edge_x_smooth"] = None
        st["distance_m_smooth"] = None
        st["last_distance_m_smooth"] = None
        st["last_time_s"] = None

        st["accepted_lane_pair"] = None
        st["pending_lane_pair"] = None
        st["pending_count"] = 0

        st["cross_count"] = 0
        st["departure_count"] = 0
        st["release_count"] = 0
        st["lane_change_count"] = 0
        st["lane_change_hold"] = 0
        st["in_right_departure"] = False

        st["last_accepted_lane_pair_for_measurement"] = None
        st["last_edge_x_for_sanity"] = None
        st["last_bbox_center_x_for_sanity"] = None

        st["lane_pair_lock_count"] = 0
        st["lane_pair_last_switch_direction"] = ""
        st["lane_revert_pending_count"] = 0

    def sanitize_right_edge(self, vehicle_id, edge_x_raw, bbox_center_x, bbox_width_px):
        """
        v4.5：对右边缘原始点做时序合理性约束。

        YOLO-seg 在部分帧会把阴影、护栏或邻近目标并入车辆 mask，
        导致 right_edge_x 单帧突然向右外跳。该异常会直接放大 distance_m 的负值。
        这里用上一帧右边缘和 bbox 中心位移预测本帧右边缘允许范围，
        超出范围时进行截断，但不改变车道线和比例尺计算。
        """
        st = self._get_state(vehicle_id)

        edge_x_raw = float(edge_x_raw)
        bbox_center_x = float(bbox_center_x)
        bbox_width_px = max(1.0, float(bbox_width_px))

        prev_edge = st.get("last_edge_x_for_sanity")
        prev_bbox_center = st.get("last_bbox_center_x_for_sanity")

        edge_x = edge_x_raw
        clamped = False
        delta_px = 0.0

        width_based_jump = EDGE_SANITY_BBOX_WIDTH_RATIO * bbox_width_px
        max_jump_px = max(EDGE_SANITY_MIN_ABS_JUMP_PX, width_based_jump)
        max_jump_px = min(float(EDGE_SANITY_MAX_ABS_JUMP_PX), float(max_jump_px))

        if EDGE_SANITY_ENABLE and prev_edge is not None and prev_bbox_center is not None:
            bbox_shift = bbox_center_x - float(prev_bbox_center)
            predicted_edge = float(prev_edge) + EDGE_SANITY_BBOX_SHIFT_GAIN * bbox_shift
            delta_px = edge_x_raw - predicted_edge

            if abs(delta_px) > max_jump_px:
                edge_x = predicted_edge + np.sign(delta_px) * max_jump_px
                clamped = True
        else:
            predicted_edge = edge_x_raw

        st["last_edge_x_for_sanity"] = float(edge_x)
        st["last_bbox_center_x_for_sanity"] = float(bbox_center_x)

        return {
            "x": float(edge_x),
            "raw_x": float(edge_x_raw),
            "predicted_x": float(predicted_edge),
            "clamped": bool(clamped),
            "delta_px": float(delta_px),
            "max_jump_px": float(max_jump_px),
        }


    def update_measurement(
        self,
        vehicle_id,
        frame_id,
        time_s,
        edge_x_raw,
        distance_m_raw,
        status_raw,
        accepted_lane_pair,
        candidate_lane_pair,
        lane_pos_ratio,
    ):
        st = self._get_state(vehicle_id)

        current_lane_pair = tuple(accepted_lane_pair) if accepted_lane_pair is not None else None
        prev_lane_pair = st.get("last_accepted_lane_pair_for_measurement")

        lane_pair_changed = (
            prev_lane_pair is not None
            and current_lane_pair is not None
            and tuple(prev_lane_pair) != tuple(current_lane_pair)
        )

        right_lane_pair_changed = (
            lane_pair_changed
            and tuple(current_lane_pair)[0] == tuple(prev_lane_pair)[1]
        )

        force_reset_smooth = False
        if RESET_SMOOTH_ON_LANE_PAIR_CHANGE and lane_pair_changed:
            force_reset_smooth = True

        if (
            st["distance_m_smooth"] is not None
            and abs(float(distance_m_raw) - float(st["distance_m_smooth"])) >= LANE_PAIR_RESET_DISTANCE_JUMP_M
            and lane_pair_changed
        ):
            force_reset_smooth = True

        if st["edge_x_smooth"] is None or force_reset_smooth:
            edge_x_smooth = float(edge_x_raw)
        else:
            edge_x_smooth = (
                RIGHT_EDGE_SMOOTH_ALPHA * float(edge_x_raw)
                + (1.0 - RIGHT_EDGE_SMOOTH_ALPHA) * float(st["edge_x_smooth"])
            )

        if st["distance_m_smooth"] is None or force_reset_smooth:
            distance_m_smooth = float(distance_m_raw)
        else:
            distance_m_smooth = (
                DISTANCE_SMOOTH_ALPHA * float(distance_m_raw)
                + (1.0 - DISTANCE_SMOOTH_ALPHA) * float(st["distance_m_smooth"])
            )

        distance_rate_mps = 0.0
        if (
            not force_reset_smooth
            and st["last_time_s"] is not None
            and st["last_distance_m_smooth"] is not None
            and float(time_s) > float(st["last_time_s"])
        ):
            dt = float(time_s) - float(st["last_time_s"])
            distance_rate_mps = (distance_m_smooth - float(st["last_distance_m_smooth"])) / dt

        if force_reset_smooth:
            # 新车道的右边界已经变成新的测距基准，旧车道的连续越线计数不能继续沿用。
            st["cross_count"] = 0
            st["departure_count"] = 0
            st["release_count"] = 0
            st["in_right_departure"] = False

        if status_raw == "CROSS":
            st["cross_count"] += 1
        else:
            st["cross_count"] = 0

        right_approach_mps = -distance_rate_mps

        if DEPARTURE_USE_CROSS_ONLY:
            departure_condition = distance_m_smooth <= CROSS_DIST_M
        else:
            departure_condition = (
                distance_m_smooth <= CROSS_DIST_M
                or (distance_m_smooth <= TOUCH_DIST_M and right_approach_mps >= RIGHT_APPROACH_SPEED_MPS)
            )

        if departure_condition:
            st["departure_count"] += 1
            st["release_count"] = 0
        elif distance_m_smooth >= DEPARTURE_RELEASE_DIST_M:
            st["release_count"] += 1
            st["departure_count"] = 0
        else:
            st["departure_count"] = 0
            st["release_count"] = 0

        if (not st["in_right_departure"]) and st["departure_count"] >= DEPARTURE_CONFIRM_FRAMES:
            st["in_right_departure"] = True

        if st["in_right_departure"] and st["release_count"] >= DEPARTURE_RELEASE_CONFIRM_FRAMES:
            st["in_right_departure"] = False

        if TRAFFIC_DIRECTION == "TOWARD_CAMERA":
            # 车辆从远往近行驶时，“行驶右侧”在图像左侧；
            # 右向换道表现为候选车道对整体向左移动。
            right_lane_change_condition = (
                st["cross_count"] >= LANE_CHANGE_CONFIRM_FRAMES
                and candidate_lane_pair is not None
                and accepted_lane_pair is not None
                and tuple(candidate_lane_pair) != tuple(accepted_lane_pair)
                and tuple(candidate_lane_pair)[1] == tuple(accepted_lane_pair)[0]
                and lane_pos_ratio <= (1.0 - LANE_CHANGE_CENTER_RATIO_TH)
            )
        else:
            # 默认：车辆从近往远行驶，“行驶右侧”在图像右侧。
            right_lane_change_condition = (
                st["cross_count"] >= LANE_CHANGE_CONFIRM_FRAMES
                and candidate_lane_pair is not None
                and accepted_lane_pair is not None
                and tuple(candidate_lane_pair) != tuple(accepted_lane_pair)
                and tuple(candidate_lane_pair)[0] == tuple(accepted_lane_pair)[1]
                and lane_pos_ratio >= LANE_CHANGE_CENTER_RATIO_TH
            )

        if right_lane_pair_changed:
            st["lane_change_hold"] = LANE_CHANGE_HOLD_FRAMES
            st["lane_change_count"] = LANE_CHANGE_CONFIRM_FRAMES
        elif right_lane_change_condition:
            st["lane_change_count"] += 1
        else:
            st["lane_change_count"] = 0

        if st["lane_change_count"] >= LANE_CHANGE_CONFIRM_FRAMES:
            st["lane_change_hold"] = max(st["lane_change_hold"], LANE_CHANGE_HOLD_FRAMES)

        lane_change_state = "KEEP"
        if st["lane_change_hold"] > 0:
            lane_change_state = "RIGHT_LANE_CHANGE"
            st["lane_change_hold"] -= 1
        elif st["in_right_departure"]:
            lane_change_state = "RIGHT_DEPARTURE"

        st["edge_x_smooth"] = edge_x_smooth
        st["distance_m_smooth"] = distance_m_smooth
        st["last_distance_m_smooth"] = distance_m_smooth
        st["last_time_s"] = float(time_s)
        st["last_accepted_lane_pair_for_measurement"] = current_lane_pair
        st["last_frame_id"] = int(frame_id)

        return {
            "vehicle_right_x_smooth": float(edge_x_smooth),
            "distance_m_smooth": float(distance_m_smooth),
            "distance_rate_mps": float(distance_rate_mps),
            "right_approach_mps": float(right_approach_mps),
            "cross_count": int(st["cross_count"]),
            "departure_count": int(st["departure_count"]),
            "release_count": int(st["release_count"]),
            "lane_change_count": int(st["lane_change_count"]),
            "lane_change_state": lane_change_state,
            "lane_pair_changed": bool(lane_pair_changed),
            "smooth_reset": bool(force_reset_smooth),
        }

    def cleanup(self, current_frame_id, max_age=150):
        remove_ids = []
        for vehicle_id, st in self.states.items():
            if st.get("last_frame_id", -1) >= 0 and current_frame_id - st["last_frame_id"] > max_age:
                remove_ids.append(vehicle_id)

        for vehicle_id in remove_ids:
            del self.states[vehicle_id]


DISTANCE_STATE = RightDistanceTemporalState()


def compute_vehicle_right_distance(vehicle, lane_map, frame_shape, frame_id=0, fps=0):
    vehicle_class = vehicle.get("class", vehicle.get("vehicle_class", vehicle.get("class_name", "")))
    with vehicle_type_parameter_context(vehicle_class):
        return _compute_vehicle_right_distance_impl(vehicle, lane_map, frame_shape, frame_id=frame_id, fps=fps)


def _compute_vehicle_right_distance_impl(vehicle, lane_map, frame_shape, frame_id=0, fps=0):
    """
    根据车辆右侧边界和车辆所在车道右边界，计算右侧距离。

    输出字段中：
    - distance_m：原始测距值，严格等于 (right_lane_x - vehicle_right_x) * meter_per_pixel；
    - distance_m_smooth：用于显示和趋势判断的平滑值；
    - lane_pos_ratio：车辆底部轮廓中心在当前车道内的位置，0 表示左边界，1 表示右边界；
    - lane_change_state：基于右距离趋势和持续越线帧数的初步换道状态。
    """
    h, w = frame_shape[:2]

    box = vehicle["box"]
    mask = vehicle.get("mask", None)

    x1, y1, x2, y2 = map(int, box)
    vehicle_id = int(vehicle.get("display_id", -1))
    bbox_center_x = 0.5 * (x1 + x2)

    # v4.3：用底部可见轮廓中心判断车辆所在车道，替代 bbox 中心。
    road_profile = get_vehicle_road_profile(mask, box)
    vehicle_center_x = float(road_profile["center_x"])
    assignment_y = int(road_profile["y"])
    ref_y = assignment_y
    time_s = frame_id / fps if fps and fps > 0 else 0.0

    lane_positions = get_sorted_lane_positions(lane_map, assignment_y, w)

    if len(lane_positions) < 2:
        return None

    candidate_left, candidate_right = locate_lane_pair_by_center(
        lane_positions,
        vehicle_center_x,
    )

    if candidate_left is None or candidate_right is None:
        return {
            "vehicle_id": vehicle_id,
            "vehicle_class": vehicle.get("class", ""),
            "vehicle_type_params": "motorcycle" if is_motorcycle_class(vehicle.get("class", "")) else "car",
            "vehicle_center_x": float(vehicle_center_x),
            "bbox_center_x": float(bbox_center_x),
            "vehicle_road_center_x": float(vehicle_center_x),
            "road_profile_source": road_profile.get("source", ""),
            "road_profile_y": int(road_profile.get("y", ref_y)),
            "road_profile_valid_rows": int(road_profile.get("valid_rows", 0)),
            "vehicle_right_x": float(x2),
            "vehicle_right_x_raw": float(x2),
            "vehicle_right_x_smooth": float(x2),
            "edge_sanity_clamped": False,
            "edge_sanity_delta_px": 0.0,
            "edge_sanity_max_jump_px": 0.0,
            "ref_y": int(ref_y),
            "lane_left_id": -1,
            "lane_right_id": -1,
            "candidate_lane_left_id": -1,
            "candidate_lane_right_id": -1,
            "left_lane_x": np.nan,
            "right_lane_x": np.nan,
            "lane_width_px": np.nan,
            "meter_per_pixel": np.nan,
            "distance_px": np.nan,
            "distance_m": np.nan,
            "distance_m_smooth": np.nan,
            "distance_rate_mps": np.nan,
            "right_approach_mps": np.nan,
            "lane_pos_ratio": np.nan,
            "right_edge_source": "out_of_lane",
            "right_edge_valid_rows": 0,
            "edge_y_low": -1,
            "edge_y_high": -1,
            "right_edge_row_width": 0,
            "status": "OUT_OF_LANE",
            "status_raw": "OUT_OF_LANE",
            "lane_change_state": "OUT_OF_LANE",
            "cross_count": 0,
        }

    # v4.9：先做测距质量门控。近景半出画大车保留检测框，但不做距离/换道判断。
    candidate_left_x_for_quality = float(candidate_left["x"])
    candidate_right_x_for_quality = float(candidate_right["x"])
    candidate_lane_width_for_quality = candidate_right_x_for_quality - candidate_left_x_for_quality
    quality = assess_measurement_quality(
        box=box,
        frame_shape=frame_shape,
        lane_width_px=candidate_lane_width_for_quality,
        assignment_y=assignment_y,
    )

    if not quality["valid"]:
        # v4.9：不可靠目标不参与距离/换道时序，避免近景大车把误差拖入后续帧。
        DISTANCE_STATE.reset_unreliable_measurement(vehicle_id)

        lane_pos_ratio_quality = (
            (vehicle_center_x - candidate_left_x_for_quality) / candidate_lane_width_for_quality
            if candidate_lane_width_for_quality > 1 else np.nan
        )
        return {
            "vehicle_id": vehicle_id,
            "vehicle_class": vehicle.get("class", ""),
            "vehicle_type_params": "motorcycle" if is_motorcycle_class(vehicle.get("class", "")) else "car",
            "vehicle_center_x": float(vehicle_center_x),
            "bbox_center_x": float(bbox_center_x),
            "vehicle_road_center_x": float(vehicle_center_x),
            "road_profile_source": road_profile.get("source", ""),
            "road_profile_y": int(road_profile.get("y", ref_y)),
            "road_profile_valid_rows": int(road_profile.get("valid_rows", 0)),
            "vehicle_center_offset_m": np.nan,
            "vehicle_right_x": float(x2),
            "vehicle_right_x_raw": float(x2),
            "vehicle_right_x_smooth": float(x2),
            "edge_sanity_clamped": False,
            "edge_sanity_delta_px": 0.0,
            "edge_sanity_pred_x": float(x2),
            "edge_sanity_max_jump_px": 0.0,
            "ref_y": int(ref_y),
            "lane_left_id": int(candidate_left["lane_id"]),
            "lane_right_id": int(candidate_right["lane_id"]),
            "candidate_lane_left_id": int(candidate_left["lane_id"]),
            "candidate_lane_right_id": int(candidate_right["lane_id"]),
            "left_lane_x": float(candidate_left_x_for_quality),
            "right_lane_x": float(candidate_right_x_for_quality),
            "lane_width_px": float(candidate_lane_width_for_quality),
            "meter_per_pixel": float(LANE_WIDTH_M / candidate_lane_width_for_quality) if candidate_lane_width_for_quality > 1 else np.nan,
            "distance_px": np.nan,
            "distance_m": np.nan,
            "distance_m_smooth": np.nan,
            "distance_rate_mps": np.nan,
            "right_approach_mps": np.nan,
            "lane_pos_ratio": float(lane_pos_ratio_quality),
            "right_edge_source": "quality_rejected",
            "right_edge_valid_rows": 0,
            "edge_y_low": -1,
            "edge_y_high": -1,
            "right_edge_row_width": 0,
            "status": "UNRELIABLE_BOX",
            "status_raw": "UNRELIABLE_BOX",
            "lane_change_state": "UNRELIABLE_BOX",
            "lane_pair_changed": False,
            "smooth_reset": False,
            "lane_assign_pending_count": 0,
            "lane_pair_lock_count": 0,
            "cross_count": 0,
            "departure_count": 0,
            "release_count": 0,
            "lane_change_count": 0,
            "measurement_valid": 0,
            "measurement_invalid_reason": quality["reason"],
            "bbox_area_ratio": float(quality["bbox_area_ratio"]),
            "bbox_height_ratio": float(quality["bbox_height_ratio"]),
            "bbox_width_ratio": float(quality["bbox_width_ratio"]),
            "bbox_width_lane_ratio": float(quality["bbox_width_lane_ratio"]),
            "bottom_clipped": int(quality["bottom_clipped"]),
            "side_clipped": int(quality["side_clipped"]),
        }

    accepted_left, accepted_right, accepted_key, candidate_key, pending_count, lane_pair_lock_count = (
        DISTANCE_STATE.select_lane_pair(
            vehicle_id,
            lane_positions,
            candidate_left,
            candidate_right,
            vehicle_center_x,
        )
    )

    if accepted_left is None or accepted_right is None:
        return None

    # v4.10：支持不同车辆行驶方向。
    # AWAY_FROM_CAMERA：车辆从近往远，行驶右侧≈图像右侧，使用右边界测距；
    # TOWARD_CAMERA：车辆从远往近，行驶右侧≈图像左侧，使用左边界测距。
    if TRAFFIC_DIRECTION == "TOWARD_CAMERA":
        measure_side = "left"
        measured_visual_side = "LEFT_AS_DRIVING_RIGHT"
        measure_lane_obj = accepted_left["lane"]
        measure_lane_id = int(accepted_left["lane_id"])
    else:
        measure_side = "right"
        measured_visual_side = "RIGHT_AS_DRIVING_RIGHT"
        measure_lane_obj = accepted_right["lane"]
        measure_lane_id = int(accepted_right["lane_id"])

    right_edge_info = get_vehicle_side_edge_near_lane(
        mask,
        box,
        measure_lane_obj,
        w,
        side=measure_side,
    )

    measure_y = int(right_edge_info["y"])
    ref_y = measure_y

    left_lane_x = float(lane_x_at_y(accepted_left["lane"], measure_y))
    right_lane_x = float(lane_x_at_y(accepted_right["lane"], measure_y))

    lane_width_px = right_lane_x - left_lane_x
    if lane_width_px <= 1:
        return None

    meter_per_pixel = LANE_WIDTH_M / lane_width_px

    vehicle_measure_x_raw = float(right_edge_info["x"])
    edge_sanity = DISTANCE_STATE.sanitize_right_edge(
        vehicle_id=vehicle_id,
        edge_x_raw=vehicle_measure_x_raw,
        bbox_center_x=bbox_center_x,
        bbox_width_px=max(1, x2 - x1),
    )
    vehicle_measure_x = float(edge_sanity["x"])

    if measure_side == "left":
        measure_lane_x = left_lane_x
        distance_px = vehicle_measure_x - left_lane_x
    else:
        measure_lane_x = right_lane_x
        distance_px = right_lane_x - vehicle_measure_x

    distance_m = distance_px * meter_per_pixel

    # 兼容旧字段名：vehicle_right_x 在 TOWARD_CAMERA 模式下实际表示“行驶右侧边缘”的图像 x。
    vehicle_right_x_raw = vehicle_measure_x_raw
    vehicle_right_x = vehicle_measure_x

    lane_center_x = 0.5 * (left_lane_x + right_lane_x)
    vehicle_center_offset_px = vehicle_center_x - lane_center_x
    vehicle_center_offset_m = vehicle_center_offset_px * meter_per_pixel
    lane_pos_ratio = (vehicle_center_x - left_lane_x) / lane_width_px

    if abs(distance_m) > MAX_VALID_DISTANCE_M:
        status_raw = "INVALID"
    else:
        status_raw = classify_distance(distance_m)

    temporal = DISTANCE_STATE.update_measurement(
        vehicle_id=vehicle_id,
        frame_id=frame_id,
        time_s=time_s,
        edge_x_raw=vehicle_right_x,
        distance_m_raw=distance_m,
        status_raw=status_raw,
        accepted_lane_pair=accepted_key,
        candidate_lane_pair=candidate_key,
        lane_pos_ratio=lane_pos_ratio,
    )

    distance_m_smooth = temporal["distance_m_smooth"]
    status = "INVALID" if abs(distance_m_smooth) > MAX_VALID_DISTANCE_M else classify_distance(distance_m_smooth)

    return {
        "vehicle_id": vehicle_id,
        "vehicle_class": vehicle.get("class", ""),
        "vehicle_type_params": "motorcycle" if is_motorcycle_class(vehicle.get("class", "")) else "car",
        "vehicle_center_x": float(vehicle_center_x),
        "bbox_center_x": float(bbox_center_x),
        "vehicle_road_center_x": float(vehicle_center_x),
        "road_profile_source": road_profile.get("source", ""),
        "road_profile_y": int(road_profile.get("y", ref_y)),
        "road_profile_valid_rows": int(road_profile.get("valid_rows", 0)),
        "vehicle_center_offset_m": float(vehicle_center_offset_m),
        "traffic_direction": TRAFFIC_DIRECTION,
        "measured_visual_side": measured_visual_side,
        "measure_lane_id": int(measure_lane_id),
        "measure_lane_x": float(measure_lane_x),
        "vehicle_measure_x": float(vehicle_measure_x),
        "vehicle_measure_x_raw": float(vehicle_measure_x_raw),
        "vehicle_measure_x_smooth": float(temporal["vehicle_right_x_smooth"]),
        "vehicle_right_x": float(vehicle_right_x),
        "vehicle_right_x_raw": float(vehicle_right_x_raw),
        "vehicle_right_x_smooth": float(temporal["vehicle_right_x_smooth"]),
        "edge_sanity_clamped": bool(edge_sanity.get("clamped", False)),
        "edge_sanity_delta_px": float(edge_sanity.get("delta_px", 0.0)),
        "edge_sanity_pred_x": float(edge_sanity.get("predicted_x", vehicle_right_x_raw)),
        "edge_sanity_max_jump_px": float(edge_sanity.get("max_jump_px", 0.0)),
        "ref_y": int(ref_y),
        "lane_left_id": int(accepted_left["lane_id"]),
        "lane_right_id": int(accepted_right["lane_id"]),
        "candidate_lane_left_id": int(candidate_left["lane_id"]),
        "candidate_lane_right_id": int(candidate_right["lane_id"]),
        "left_lane_x": float(left_lane_x),
        "right_lane_x": float(right_lane_x),
        "lane_width_px": float(lane_width_px),
        "meter_per_pixel": float(meter_per_pixel),
        "distance_px": float(distance_px),
        "distance_m": float(distance_m),
        "distance_m_smooth": float(distance_m_smooth),
        "distance_rate_mps": float(temporal["distance_rate_mps"]),
        "right_approach_mps": float(temporal["right_approach_mps"]),
        "lane_pos_ratio": float(lane_pos_ratio),
        "right_edge_source": right_edge_info["source"],
        "right_edge_valid_rows": int(right_edge_info["valid_rows"]),
        "edge_y_low": int(right_edge_info["y_low"]),
        "edge_y_high": int(right_edge_info["y_high"]),
        "right_edge_row_width": int(right_edge_info.get("row_width", 0)),
        "status": status,
        "status_raw": status_raw,
        "lane_change_state": temporal["lane_change_state"],
        "lane_pair_changed": bool(temporal.get("lane_pair_changed", False)),
        "smooth_reset": bool(temporal.get("smooth_reset", False)),
        "lane_assign_pending_count": int(pending_count),
        "lane_pair_lock_count": int(lane_pair_lock_count),
        "cross_count": int(temporal["cross_count"]),
        "departure_count": int(temporal.get("departure_count", 0)),
        "release_count": int(temporal.get("release_count", 0)),
        "lane_change_count": int(temporal.get("lane_change_count", 0)),
        "measurement_valid": 1,
        "measurement_invalid_reason": "OK",
        "bbox_area_ratio": float(((x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))),
        "bbox_height_ratio": float((y2 - y1) / max(1.0, float(h))),
        "bbox_width_ratio": float((x2 - x1) / max(1.0, float(w))),
        "bbox_width_lane_ratio": float((x2 - x1) / max(1.0, float(lane_width_px))),
        "bottom_clipped": int(y2 >= h - MEASUREMENT_BOTTOM_CLIP_MARGIN_PX),
        "side_clipped": int(x1 <= MEASUREMENT_SIDE_CLIP_MARGIN_PX or x2 >= w - MEASUREMENT_SIDE_CLIP_MARGIN_PX),
    }


# =========================================================
# 可视化
# =========================================================

def get_vehicle_box_color(item):
    """按目标类别返回车辆框颜色。OpenCV 使用 BGR，摩托车用红色。"""
    vehicle_class = str(
        item.get("class", item.get("vehicle_class", item.get("class_name", "")))
    ).lower()

    if vehicle_class == "motorcycle":
        return MOTORCYCLE_BOX_COLOR

    return VEHICLE_BOX_COLOR

def draw_vehicle_top_layer(frame, boxes, contours):
    out = frame.copy()

    if SHOW_VEHICLE_MASK_EDGE:
        for pts in contours:
            cv2.polylines(out, [pts], True, VEHICLE_EDGE_COLOR, 1, cv2.LINE_AA)

    if SHOW_VEHICLE_BOX:
        for item in boxes:
            x1, y1, x2, y2 = item["box"]
            conf = item["conf"]
            display_id = item.get("display_id", None)

            box_color = get_vehicle_box_color(item)

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 1)

            if display_id is not None:
                label = f"ID{display_id} {conf:.2f}"
            else:
                label = f"{conf:.2f}"

            cv2.putText(
                out,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA,
            )

    return out


def draw_distance_overlay(frame, distance_results):
    out = frame.copy()

    for item in distance_results:
        if item is None:
            continue

        status = item["status"]

        if status in {"OUT_OF_LANE", "INVALID", "UNRELIABLE_BOX"}:
            continue

        color = status_to_color(status)

        # v4.2：测距点用原始本帧右边缘，避免平滑后点漂出车辆轮廓。
        # 数值仍显示 distance_m_smooth，兼顾视觉连接点和趋势稳定性。
        vehicle_measure_x = int(round(item.get("vehicle_measure_x", item.get("vehicle_right_x"))))
        measure_lane_x = int(round(item.get("measure_lane_x", item.get("right_lane_x"))))
        ref_y = int(item["ref_y"])
        vehicle_id = int(item["vehicle_id"])
        distance_m = float(item.get("distance_m_smooth", item["distance_m"]))
        lane_change_state = item.get("lane_change_state", "KEEP")
        measured_visual_side = item.get("measured_visual_side", "RIGHT_AS_DRIVING_RIGHT")

        p_vehicle = (vehicle_measure_x, ref_y)
        p_lane = (measure_lane_x, ref_y)

        if SHOW_DISTANCE_LINE:
            cv2.circle(out, p_vehicle, 4, color, -1, cv2.LINE_AA)
            cv2.circle(out, p_lane, 4, color, -1, cv2.LINE_AA)

            # 测距线固定为同一 ref_y 的水平线，确保线段两端分别落在车辆右边界和右车道线。
            cv2.line(out, p_vehicle, p_lane, color, 2, cv2.LINE_AA)

            # 两端短竖线用于检查连接点，不影响距离定义。
            cv2.line(out, (vehicle_measure_x, ref_y - 8), (vehicle_measure_x, ref_y + 8), color, 2, cv2.LINE_AA)
            cv2.line(out, (measure_lane_x, ref_y - 8), (measure_lane_x, ref_y + 8), color, 2, cv2.LINE_AA)

        if SHOW_DISTANCE_TEXT:
            side_label = "DrivingRight"
            if measured_visual_side == "LEFT_AS_DRIVING_RIGHT":
                side_label = "DrivingRight(L)"
            elif measured_visual_side == "RIGHT_AS_DRIVING_RIGHT":
                side_label = "DrivingRight(R)"
            text = f"ID{vehicle_id} {side_label} {distance_m:.2f}m {status}"
            if lane_change_state != "KEEP":
                text += f" {lane_change_state}"

            tx = min(vehicle_measure_x, measure_lane_x)
            ty = max(25, ref_y - 10)

            cv2.putText(
                out,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

    return out


# =========================================================
# 单帧渲染
# =========================================================
def render_frame(frame, lane_layers, vehicle_mask, boxes, contours, distance_results):
    roi_mask = lane_layers["roi_mask"]

    hard_vehicle_mask, soft_vehicle_mask = make_soft_vehicle_mask(vehicle_mask)

    non_vehicle_mask = cv2.bitwise_and(
        roi_mask,
        cv2.bitwise_not(hard_vehicle_mask),
    )

    normal_cont_mask = cv2.bitwise_and(
        lane_layers["continuous_mask"],
        non_vehicle_mask,
    )
    normal_paint_mask = cv2.bitwise_and(
        lane_layers["paint_mask"],
        non_vehicle_mask,
    )

    normal_k = ensure_odd(NORMAL_MASK_BLUR_KSIZE)
    normal_cont_soft = cv2.GaussianBlur(normal_cont_mask, (normal_k, normal_k), 0)
    normal_paint_soft = cv2.GaussianBlur(normal_paint_mask, (normal_k, normal_k), 0)

    ghost_k = ensure_odd(GHOST_MASK_BLUR_KSIZE)
    ghost_line_soft = cv2.GaussianBlur(
        lane_layers["ghost_mask"],
        (ghost_k, ghost_k),
        0,
    )

    occluded_soft = cv2.bitwise_and(
        soft_vehicle_mask,
        ghost_line_soft,
    )

    occluded_soft = (
        occluded_soft.astype(np.float32) * 0.65
    ).clip(0, 255).astype(np.uint8)

    result = frame.copy()

    result = blend_color_by_soft_mask(
        result,
        CONTINUOUS_COLOR,
        normal_cont_soft,
        CONTINUOUS_ALPHA,
    )

    result = blend_color_by_soft_mask(
        result,
        PAINT_COLOR,
        normal_paint_soft,
        PAINT_ALPHA,
    )

    result = blend_color_by_soft_mask(
        result,
        GHOST_COLOR,
        occluded_soft,
        GHOST_ALPHA,
    )

    result = draw_distance_overlay(result, distance_results)

    if DRAW_ROI_BORDER:
        cv2.polylines(result, [lane_layers["roi_pts"]], True, (0, 255, 0), 1)

    if SHOW_DEBUG_TEXT:
        cv2.putText(
            result,
            "Right Distance V4.9",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    result = draw_vehicle_top_layer(result, boxes, contours)

    return result


# =========================================================
# CSV
# =========================================================
def make_csv_writer(csv_path):
    f = open(csv_path, "w", newline="", encoding="utf-8-sig")

    fieldnames = [
        "frame_id",
        "time_s",
        "vehicle_id",
        "vehicle_class",
        "vehicle_type_params",
        "vehicle_center_x",
        "bbox_center_x",
        "vehicle_road_center_x",
        "road_profile_source",
        "road_profile_y",
        "road_profile_valid_rows",
        "vehicle_center_offset_m",
        "traffic_direction",
        "measured_visual_side",
        "measure_lane_id",
        "measure_lane_x",
        "vehicle_measure_x",
        "vehicle_measure_x_raw",
        "vehicle_measure_x_smooth",
        "vehicle_right_x",
        "vehicle_right_x_raw",
        "vehicle_right_x_smooth",
        "edge_sanity_clamped",
        "edge_sanity_delta_px",
        "edge_sanity_pred_x",
        "edge_sanity_max_jump_px",
        "ref_y",
        "lane_left_id",
        "lane_right_id",
        "candidate_lane_left_id",
        "candidate_lane_right_id",
        "left_lane_x",
        "right_lane_x",
        "lane_width_px",
        "meter_per_pixel",
        "distance_px",
        "distance_m",
        "distance_m_smooth",
        "distance_rate_mps",
        "right_approach_mps",
        "lane_pos_ratio",
        "right_edge_source",
        "right_edge_valid_rows",
        "edge_y_low",
        "edge_y_high",
        "right_edge_row_width",
        "status",
        "status_raw",
        "lane_change_state",
        "lane_pair_changed",
        "smooth_reset",
        "lane_assign_pending_count",
        "lane_pair_lock_count",
        "cross_count",
        "departure_count",
        "release_count",
        "lane_change_count",
        "measurement_valid",
        "measurement_invalid_reason",
        "bbox_area_ratio",
        "bbox_height_ratio",
        "bbox_width_ratio",
        "bbox_width_lane_ratio",
        "bottom_clipped",
        "side_clipped",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def write_distance_rows(writer, frame_id, fps, distance_results):
    time_s = frame_id / fps if fps > 0 else 0.0

    for item in distance_results:
        if item is None:
            continue

        row = {
            "frame_id": int(frame_id),
            "time_s": round(float(time_s), 4),
            "vehicle_id": item.get("vehicle_id", -1),
            "vehicle_class": item.get("vehicle_class", ""),
            "vehicle_type_params": item.get("vehicle_type_params", ""),
            "vehicle_center_x": round_float(item.get("vehicle_center_x")),
            "bbox_center_x": round_float(item.get("bbox_center_x")),
            "vehicle_road_center_x": round_float(item.get("vehicle_road_center_x")),
            "road_profile_source": item.get("road_profile_source", ""),
            "road_profile_y": item.get("road_profile_y", -1),
            "road_profile_valid_rows": item.get("road_profile_valid_rows", 0),
            "vehicle_center_offset_m": round_float(item.get("vehicle_center_offset_m"), 4),
            "traffic_direction": item.get("traffic_direction", ""),
            "measured_visual_side": item.get("measured_visual_side", ""),
            "measure_lane_id": item.get("measure_lane_id", -1),
            "measure_lane_x": round_float(item.get("measure_lane_x")),
            "vehicle_measure_x": round_float(item.get("vehicle_measure_x")),
            "vehicle_measure_x_raw": round_float(item.get("vehicle_measure_x_raw")),
            "vehicle_measure_x_smooth": round_float(item.get("vehicle_measure_x_smooth")),
            "vehicle_right_x": round_float(item.get("vehicle_right_x")),
            "vehicle_right_x_raw": round_float(item.get("vehicle_right_x_raw")),
            "vehicle_right_x_smooth": round_float(item.get("vehicle_right_x_smooth")),
            "edge_sanity_clamped": int(bool(item.get("edge_sanity_clamped", False))),
            "edge_sanity_delta_px": round_float(item.get("edge_sanity_delta_px")),
            "edge_sanity_pred_x": round_float(item.get("edge_sanity_pred_x")),
            "edge_sanity_max_jump_px": round_float(item.get("edge_sanity_max_jump_px")),
            "ref_y": item.get("ref_y", -1),
            "lane_left_id": item.get("lane_left_id", -1),
            "lane_right_id": item.get("lane_right_id", -1),
            "candidate_lane_left_id": item.get("candidate_lane_left_id", -1),
            "candidate_lane_right_id": item.get("candidate_lane_right_id", -1),
            "left_lane_x": round_float(item.get("left_lane_x")),
            "right_lane_x": round_float(item.get("right_lane_x")),
            "lane_width_px": round_float(item.get("lane_width_px")),
            "meter_per_pixel": round_float(item.get("meter_per_pixel"), 6),
            "distance_px": round_float(item.get("distance_px")),
            "distance_m": round_float(item.get("distance_m"), 4),
            "distance_m_smooth": round_float(item.get("distance_m_smooth"), 4),
            "distance_rate_mps": round_float(item.get("distance_rate_mps"), 4),
            "right_approach_mps": round_float(item.get("right_approach_mps"), 4),
            "lane_pos_ratio": round_float(item.get("lane_pos_ratio"), 4),
            "right_edge_source": item.get("right_edge_source", ""),
            "right_edge_valid_rows": item.get("right_edge_valid_rows", 0),
            "edge_y_low": item.get("edge_y_low", -1),
            "edge_y_high": item.get("edge_y_high", -1),
            "right_edge_row_width": item.get("right_edge_row_width", 0),
            "status": item.get("status", ""),
            "status_raw": item.get("status_raw", ""),
            "lane_change_state": item.get("lane_change_state", ""),
            "lane_pair_changed": int(bool(item.get("lane_pair_changed", False))),
            "smooth_reset": int(bool(item.get("smooth_reset", False))),
            "lane_assign_pending_count": item.get("lane_assign_pending_count", 0),
            "lane_pair_lock_count": item.get("lane_pair_lock_count", 0),
            "cross_count": item.get("cross_count", 0),
            "departure_count": item.get("departure_count", 0),
            "release_count": item.get("release_count", 0),
            "lane_change_count": item.get("lane_change_count", 0),
            "measurement_valid": item.get("measurement_valid", 1),
            "measurement_invalid_reason": item.get("measurement_invalid_reason", "OK"),
            "bbox_area_ratio": round_float(item.get("bbox_area_ratio"), 6),
            "bbox_height_ratio": round_float(item.get("bbox_height_ratio"), 6),
            "bbox_width_ratio": round_float(item.get("bbox_width_ratio"), 6),
            "bbox_width_lane_ratio": round_float(item.get("bbox_width_lane_ratio"), 4),
            "bottom_clipped": item.get("bottom_clipped", 0),
            "side_clipped": item.get("side_clipped", 0),
        }

        writer.writerow(row)


def round_float(v, ndigits=3):
    try:
        if v is None or not np.isfinite(float(v)):
            return ""
        return round(float(v), ndigits)
    except Exception:
        return ""


# =========================================================
# 主程序
# =========================================================

def apply_runtime_video_scaling(width, height, fps, frame_count):
    """
    根据当前视频分辨率和帧率自动调整像素阈值、YOLO 输入尺寸和帧数阈值。
    只在 main() 中读取视频信息后调用一次。
    """
    global YOLO_IMGSZ
    global SOFT_MASK_BLUR_KSIZE, NORMAL_MASK_BLUR_KSIZE, GHOST_MASK_BLUR_KSIZE
    global VEHICLE_MASK_DILATE_X, VEHICLE_MASK_DILATE_Y
    global RIGHT_EDGE_BAND_MIN, RIGHT_EDGE_BAND_MAX
    global CUSTOM_TRACK_MAX_MISSING, CUSTOM_TRACK_MIN_HITS
    global MIN_ROI_INTERSECTION
    global MOTORCYCLE_RIGHT_EDGE_BAND_MIN, MOTORCYCLE_RIGHT_EDGE_BAND_MAX
    global MOTORCYCLE_RIGHT_EDGE_ROW_MIN_PIXELS, MOTORCYCLE_RIGHT_EDGE_MIN_VALID_ROWS
    global MOTORCYCLE_MIN_ROI_INTERSECTION
    global MOTORCYCLE_MEASUREMENT_BOTTOM_CLIP_MARGIN_PX, MOTORCYCLE_MEASUREMENT_SIDE_CLIP_MARGIN_PX
    global MOTORCYCLE_MEASUREMENT_LOW_REF_MARGIN_PX
    global MOTORCYCLE_CUSTOM_TRACK_MAX_MISSING, MOTORCYCLE_CUSTOM_TRACK_MIN_HITS
    global MOTORCYCLE_EDGE_SANITY_MIN_ABS_JUMP_PX, MOTORCYCLE_EDGE_SANITY_MAX_ABS_JUMP_PX
    global EDGE_SANITY_MIN_ABS_JUMP_PX, EDGE_SANITY_MAX_ABS_JUMP_PX
    global MEASUREMENT_BOTTOM_CLIP_MARGIN_PX, MEASUREMENT_SIDE_CLIP_MARGIN_PX
    global MEASUREMENT_MIN_VALID_LANE_WIDTH_PX, MEASUREMENT_LOW_REF_MARGIN_PX
    global LANE_ASSIGN_CONFIRM_FRAMES
    global DEPARTURE_CONFIRM_FRAMES, DEPARTURE_RELEASE_CONFIRM_FRAMES
    global LANE_CHANGE_CONFIRM_FRAMES, LANE_CHANGE_HOLD_FRAMES
    global LANE_PAIR_LOCK_AFTER_RIGHT_CHANGE_FRAMES, LANE_PAIR_LOCK_LEFT_REVERT_CONFIRM_FRAMES
    global CONTINUOUS_THICKNESS, PAINT_THICKNESS, GHOST_THICKNESS

    YOLO_IMGSZ = cfg.auto_yolo_imgsz(width, height)

    SOFT_MASK_BLUR_KSIZE = cfg.scaled_odd(17, width, height, min_value=5)
    NORMAL_MASK_BLUR_KSIZE = cfg.scaled_odd(3, width, height, min_value=3)
    GHOST_MASK_BLUR_KSIZE = cfg.scaled_odd(5, width, height, min_value=3)

    VEHICLE_MASK_DILATE_X = cfg.scale_px(15, width, height, min_value=5)
    VEHICLE_MASK_DILATE_Y = cfg.scale_px(9, width, height, min_value=3)

    RIGHT_EDGE_BAND_MIN = cfg.scale_px(6, width, height, min_value=3)
    RIGHT_EDGE_BAND_MAX = cfg.scale_px(28, width, height, min_value=8)

    MOTORCYCLE_RIGHT_EDGE_BAND_MIN = cfg.scale_px(3, width, height, min_value=2)
    MOTORCYCLE_RIGHT_EDGE_BAND_MAX = cfg.scale_px(20, width, height, min_value=6)
    MOTORCYCLE_RIGHT_EDGE_ROW_MIN_PIXELS = cfg.scale_px(2, width, height, min_value=1)
    MOTORCYCLE_RIGHT_EDGE_MIN_VALID_ROWS = 2

    MIN_ROI_INTERSECTION = cfg.scale_px(20, width, height, min_value=8)
    MOTORCYCLE_MIN_ROI_INTERSECTION = cfg.scale_px(8, width, height, min_value=4)

    EDGE_SANITY_MIN_ABS_JUMP_PX = cfg.scale_px(28, width, height, min_value=12)
    EDGE_SANITY_MAX_ABS_JUMP_PX = cfg.scale_px(58, width, height, min_value=25)
    MOTORCYCLE_EDGE_SANITY_MIN_ABS_JUMP_PX = cfg.scale_px(18, width, height, min_value=8)
    MOTORCYCLE_EDGE_SANITY_MAX_ABS_JUMP_PX = cfg.scale_px(42, width, height, min_value=18)

    MEASUREMENT_BOTTOM_CLIP_MARGIN_PX = cfg.scale_px(16, width, height, min_value=6)
    MEASUREMENT_SIDE_CLIP_MARGIN_PX = cfg.scale_px(4, width, height, min_value=2)
    MEASUREMENT_MIN_VALID_LANE_WIDTH_PX = float(cfg.scale_px(35, width, height, min_value=15))
    MEASUREMENT_LOW_REF_MARGIN_PX = cfg.scale_px(22, width, height, min_value=8)

    MOTORCYCLE_MEASUREMENT_BOTTOM_CLIP_MARGIN_PX = cfg.scale_px(10, width, height, min_value=4)
    MOTORCYCLE_MEASUREMENT_SIDE_CLIP_MARGIN_PX = cfg.scale_px(2, width, height, min_value=1)
    MOTORCYCLE_MEASUREMENT_LOW_REF_MARGIN_PX = cfg.scale_px(12, width, height, min_value=5)

    CUSTOM_TRACK_MAX_MISSING = cfg.frames_from_seconds(0.35, fps, min_frames=6)
    CUSTOM_TRACK_MIN_HITS = max(1, cfg.frames_from_seconds(0.04, fps, min_frames=1))
    MOTORCYCLE_CUSTOM_TRACK_MAX_MISSING = cfg.frames_from_seconds(0.45, fps, min_frames=8)
    MOTORCYCLE_CUSTOM_TRACK_MIN_HITS = 1

    LANE_ASSIGN_CONFIRM_FRAMES = cfg.frames_from_seconds(0.06, fps, min_frames=2)
    DEPARTURE_CONFIRM_FRAMES = cfg.frames_from_seconds(0.08, fps, min_frames=2)
    DEPARTURE_RELEASE_CONFIRM_FRAMES = cfg.frames_from_seconds(0.10, fps, min_frames=2)
    LANE_CHANGE_CONFIRM_FRAMES = cfg.frames_from_seconds(0.16, fps, min_frames=3)
    LANE_CHANGE_HOLD_FRAMES = cfg.frames_from_seconds(0.36, fps, min_frames=5)
    LANE_PAIR_LOCK_AFTER_RIGHT_CHANGE_FRAMES = cfg.frames_from_seconds(0.70, fps, min_frames=8)
    LANE_PAIR_LOCK_LEFT_REVERT_CONFIRM_FRAMES = cfg.frames_from_seconds(0.16, fps, min_frames=3)

    CONTINUOUS_THICKNESS = cfg.scale_px(2, width, height, min_value=1)
    PAINT_THICKNESS = cfg.scale_px(2, width, height, min_value=1)
    GHOST_THICKNESS = cfg.scale_px(1, width, height, min_value=1)


def main():
    root_dir = cfg.project_root_from_file(__file__)

    video_path = cfg.video_path(root_dir)
    lane_map_path = cfg.lane_map_path(root_dir)
    model_path = cfg.model_path(root_dir)

    output_video_path = cfg.outputs_dir(root_dir) / cfg.RIGHT_DISTANCE_VIDEO_NAME
    output_csv_path = cfg.outputs_dir(root_dir) / cfg.RIGHT_DISTANCE_CSV_NAME

    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频：{video_path}")
    if not lane_map_path.exists():
        raise FileNotFoundError(f"找不到稳定车道地图：{lane_map_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型：{model_path}")

    lane_map = load_lane_map(lane_map_path)

    print("正在加载 YOLO 分割模型...")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    apply_runtime_video_scaling(width, height, fps, frame_count)

    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.START_FRAME)
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("无法读取第一帧")

    lane_layers = build_lane_layers(first_frame.shape, lane_map)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{output_video_path}")

    csv_file, csv_writer = make_csv_writer(output_csv_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.START_FRAME)
    total_to_process = cfg.process_frame_count(fps, frame_count, cfg.START_FRAME)

    print("========== 车辆右侧到右车道线距离计算 V4.9.8 ==========")
    print(f"输入视频: {video_path}")
    print(f"稳定车道地图: {lane_map_path}")
    print(f"YOLO 模型: {model_path}")
    print(f"输出视频: {output_video_path}")
    print(f"输出 CSV: {output_csv_path}")
    print(f"处理帧数: {total_to_process}")
    print(f"车道宽度假设: {LANE_WIDTH_M:.2f} m")
    print("距离定义：distance_m = 右车道线x - 车辆右侧x；小于 0 表示越过右侧车道线。")
    print("v4.5：在 v4.4 基础上加入右边缘时序异常跳变约束，抑制 mask 粘连/毛刺导致的测距点突然外跳。")

    processed = 0
    frame_id = cfg.START_FRAME

    while processed < total_to_process:
        ok, frame = cap.read()
        if not ok:
            break

        vehicle_mask, boxes, contours = detect_vehicle_mask_and_boxes(
            model,
            frame,
            lane_layers["roi_mask"],
        )

        distance_results = []

        for vehicle in boxes:
            dist_item = compute_vehicle_right_distance(
                vehicle,
                lane_map,
                frame.shape,
                frame_id=frame_id,
                fps=fps,
            )

            if dist_item is not None:
                distance_results.append(dist_item)

        write_distance_rows(csv_writer, frame_id, fps, distance_results)

        result = render_frame(
            frame,
            lane_layers,
            vehicle_mask,
            boxes,
            contours,
            distance_results,
        )

        writer.write(result)

        DISTANCE_STATE.cleanup(frame_id)

        processed += 1
        frame_id += 1

        if processed % 50 == 0:
            print(f"已处理 {processed}/{total_to_process} 帧")

    cap.release()
    writer.release()
    csv_file.close()

    print("完成")
    print(f"输出视频: {output_video_path}")
    print(f"输出 CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
