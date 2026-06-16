# -*- coding: utf-8 -*-
r"""
export_vehicle_trajectory_xy.py

功能：
1. 读取 outputs/vehicle_distance.csv 中每辆车的逐帧轨迹坐标；
2. 读取 configs/roi_config.json 中手动画出的 ROI 多边形；
3. 以 ROI 外接矩形左上角作为 XY 坐标原点 (0, 0)；
4. 用 ROI polygon 判断轨迹点是否真实位于 ROI 内；
5. 主 Excel 表格、车辆汇总、主 XY 图、每车单独图只使用 ROI 内部轨迹点；
6. 不对坐标做裁剪、不强行把点投影到 ROI 边界；
7. 额外保留“原始全部轨迹坐标”sheet，便于检查被过滤掉的 ROI 外点。

运行方式：
    cd D:\zhuanli
    python src\export_vehicle_trajectory_xy.py

默认输入：
    outputs\vehicle_distance.csv
    configs\roi_config.json
    outputs\video_meta.json（可选）

默认输出：
    outputs\vehicle_trajectory_roi_bbox_topleft_xy.xlsx
    outputs\vehicle_trajectory_roi_bbox_topleft_xy_overview.png
    outputs\vehicle_trajectory_roi_bbox_topleft_by_id\IDxxx_roi_bbox_topleft_xy.png

坐标定义：
- trajectory_x_image：原始图像 x 坐标，图像左上角为原点，x 向右；
- trajectory_y_image：原始图像 y 坐标，图像左上角为原点，y 向下；
- roi_origin_x_image / roi_origin_y_image：ROI 外接矩形左上角在图像坐标中的位置；
- trajectory_x_roi：以 ROI 外接矩形左上角为原点后的 x 坐标，正方向为图像向右；
- trajectory_y_roi：以 ROI 外接矩形左上角为原点后的 y 坐标，正方向为图像向下；
- inside_roi：轨迹点是否位于 ROI polygon 内部或边界上。

注意：
- 本脚本不是“裁剪轨迹”，而是按真实 ROI 多边形过滤数据；
- ROI 外点不会进入主图和主汇总，但会保留在 Excel 的“原始全部轨迹坐标”sheet 中；
- 如果你希望主表也保留所有点，可把 FILTER_MAIN_OUTPUT_TO_ROI_INSIDE 改为 False。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 用户可调参数
# =========================

# 是否只导出 measurement_valid 为真值的记录。一般先 False，保持和测距 CSV 一致。
EXPORT_ONLY_VALID_MEASUREMENT = False

# 主表、主图、每车单图是否只使用 ROI 内部点。按你的要求，这里默认 True。
FILTER_MAIN_OUTPUT_TO_ROI_INSIDE = True

# 每辆车单独画图至少需要几个 ROI 内轨迹点。
MIN_POINTS_FOR_SINGLE_ID_PLOT = 2

# 是否为每辆车写一个单独 sheet。
WRITE_EACH_VEHICLE_SHEET = True
MAX_VEHICLE_SHEETS = 80

# ROI 原点模式：
# BOUNDING_TOP_LEFT：使用 ROI 外接矩形左上角，即 (min_x, min_y)。
# 这样只要轨迹点真实在 ROI 内，X/Y 一般不会出现负数。
# VERTEX_TOP_LEFT：保留为可选调试模式，不建议用于最终图。
ROI_ORIGIN_MODE = "BOUNDING_TOP_LEFT"

# 如果一辆车 ROI 内点太少，是否仍保留在汇总表中。False 更干净。
KEEP_VEHICLE_WITH_SINGLE_POINT_IN_SUMMARY = True

OVERVIEW_FIGSIZE = (12, 7)
SINGLE_ID_FIGSIZE = (7, 5)
DPI = 160
LINE_WIDTH = 1.4
POINT_SIZE = 10
SINGLE_ID_LINE_WIDTH = 2.0
SINGLE_ID_POINT_SIZE = 18

# VERTEX_TOP_LEFT 模式下，用 y 最小的一组点估计 ROI 顶部边界，再取其中 x 最小的点。
# 一般手动画 ROI 是四边形，取 2 个顶点即可。
ROI_TOP_POINT_COUNT = 2


# =========================
# 路径与基础读取
# =========================

def get_project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        if here.parent.name.lower() == "src":
            return here.parent.parent
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    if cwd.name.lower() == "src":
        return cwd.parent
    return cwd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_video_meta(root: Path) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    candidates = [
        root / "outputs" / "video_meta.json",
        root / "video_meta.json",
    ]
    for p in candidates:
        meta = read_json_if_exists(p)
        if not meta:
            continue
        width = meta.get("width") or meta.get("image_width")
        height = meta.get("height") or meta.get("image_height")
        fps = meta.get("fps")
        return (
            int(width) if width is not None else None,
            int(height) if height is not None else None,
            float(fps) if fps is not None else None,
        )
    return None, None, None


def _extract_roi_points_from_json(data: dict) -> Optional[List[List[float]]]:
    """兼容常见 ROI 配置字段。"""
    for key in ["roi_polygon", "polygon", "points", "roi_points"]:
        if key in data and data[key] is not None:
            return data[key]

    # 有些配置可能写成 {"roi": {"points": [...]}}
    if isinstance(data.get("roi"), dict):
        for key in ["roi_polygon", "polygon", "points", "roi_points"]:
            if key in data["roi"] and data["roi"][key] is not None:
                return data["roi"][key]

    return None


def read_roi_config(root: Path, roi_path_arg: Optional[str] = None) -> Tuple[Optional[np.ndarray], str]:
    candidates = []
    if roi_path_arg:
        p = Path(roi_path_arg)
        candidates.append(p if p.is_absolute() else root / p)

    candidates.extend([
        root / "configs" / "roi_config.json",
        root / "roi_config.json",
    ])

    for p in candidates:
        data = read_json_if_exists(p)
        if not data:
            continue
        poly = _extract_roi_points_from_json(data)
        if poly is None:
            continue
        arr = np.array(poly, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
            return arr[:, :2], str(p)

    return None, "not_found"


# =========================
# ROI 原点与几何判断
# =========================

def compute_roi_top_left_origin(
    roi_polygon: Optional[np.ndarray],
    mode: str = ROI_ORIGIN_MODE,
) -> Tuple[float, float, str, Optional[np.ndarray]]:
    """
    返回 ROI 原点：origin_x_image, origin_y_image。

    VERTEX_TOP_LEFT：
    - 先取 y 最小的几个顶点作为 ROI 顶部边界候选；
    - 再取其中 x 最小的点作为 ROI 外接矩形左上角。

    BOUNDING_TOP_LEFT：
    - origin_x = ROI 所有顶点 x 的最小值；
    - origin_y = ROI 所有顶点 y 的最小值；
    - 这是 ROI 外接矩形左上角，用于最终轨迹图，能够避免倾斜 ROI 顶点导致的负坐标。
    """
    mode = str(mode).strip().upper()

    if roi_polygon is None or len(roi_polygon) < 3:
        return 0.0, 0.0, "fallback_image_top_left", None

    pts = np.asarray(roi_polygon, dtype=float)

    if mode == "BOUNDING_TOP_LEFT":
        origin_x = float(np.nanmin(pts[:, 0]))
        origin_y = float(np.nanmin(pts[:, 1]))
        selected = np.array([[origin_x, origin_y]], dtype=float)
        return origin_x, origin_y, "roi_bounding_box_top_left", selected

    # 默认：ROI 顶部边界左端顶点。
    n = min(max(1, ROI_TOP_POINT_COUNT), len(pts))
    top_order = np.argsort(pts[:, 1])[:n]
    top_candidates = pts[top_order]
    left_idx = int(np.argmin(top_candidates[:, 0]))
    origin_x = float(top_candidates[left_idx, 0])
    origin_y = float(top_candidates[left_idx, 1])
    return origin_x, origin_y, "roi_vertex_top_left", top_candidates


def convert_to_roi_xy(
    x_image: pd.Series,
    y_image: pd.Series,
    origin_x: float,
    origin_y: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    以 ROI 外接矩形左上角为原点：
    x 正方向：图像向右；
    y 正方向：图像向下。
    """
    x_roi = x_image.astype(float) - float(origin_x)
    y_roi = y_image.astype(float) - float(origin_y)
    return x_roi, y_roi


def convert_roi_polygon_to_relative(
    roi_polygon: Optional[np.ndarray],
    origin_x: float,
    origin_y: float,
) -> Optional[np.ndarray]:
    if roi_polygon is None:
        return None
    x = pd.Series(roi_polygon[:, 0], dtype=float)
    y = pd.Series(roi_polygon[:, 1], dtype=float)
    xr, yr = convert_to_roi_xy(x, y, origin_x, origin_y)
    return np.column_stack([xr.to_numpy(), yr.to_numpy()])


def point_inside_roi_polygon(x: float, y: float, roi_polygon: Optional[np.ndarray]) -> Optional[bool]:
    if roi_polygon is None or len(roi_polygon) < 3:
        return None
    pts = np.asarray(roi_polygon, dtype=np.float32)
    val = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
    return bool(val >= 0)


# =========================
# 数据处理
# =========================

def choose_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float) != 0
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "ok", "valid"])


def infer_image_size(df: pd.DataFrame, x_col: str, y_col: str, root: Path) -> Tuple[int, int, Optional[float], str]:
    meta_w, meta_h, fps = read_video_meta(root)
    if meta_w and meta_h:
        return int(meta_w), int(meta_h), fps, "video_meta.json"

    max_x = pd.to_numeric(df[x_col], errors="coerce").max()
    max_y = pd.to_numeric(df[y_col], errors="coerce").max()
    width = int(max(1280, math.ceil(float(max_x) + 80))) if pd.notna(max_x) else 1280
    height = int(max(720, math.ceil(float(max_y) + 80))) if pd.notna(max_y) else 720
    return width, height, fps, "estimated_from_csv"


def build_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if points.empty:
        return pd.DataFrame()

    for vid, g in points.groupby("vehicle_id", sort=True):
        g = g.sort_values("frame_id")
        if len(g) == 1 and not KEEP_VEHICLE_WITH_SINGLE_POINT_IN_SUMMARY:
            continue

        cls = str(g["vehicle_class"].mode().iloc[0]) if not g["vehicle_class"].dropna().empty else "vehicle"
        dx = g["trajectory_x_roi"].diff()
        dy = g["trajectory_y_roi"].diff()
        path_len_px = np.sqrt(dx * dx + dy * dy).sum(skipna=True)

        duration_s = np.nan
        if "time_s" in g.columns and g["time_s"].notna().sum() >= 2:
            duration_s = float(g["time_s"].max() - g["time_s"].min())

        row: Dict[str, Any] = {
            "vehicle_id": int(vid),
            "vehicle_class": cls,
            "roi_inside_point_count": int(len(g)),
            "start_frame": int(g["frame_id"].min()),
            "end_frame": int(g["frame_id"].max()),
            "duration_s": duration_s,
            "start_x_roi": float(g["trajectory_x_roi"].iloc[0]),
            "start_y_roi": float(g["trajectory_y_roi"].iloc[0]),
            "end_x_roi": float(g["trajectory_x_roi"].iloc[-1]),
            "end_y_roi": float(g["trajectory_y_roi"].iloc[-1]),
            "start_x_image": float(g["trajectory_x_image"].iloc[0]),
            "start_y_image": float(g["trajectory_y_image"].iloc[0]),
            "end_x_image": float(g["trajectory_x_image"].iloc[-1]),
            "end_y_image": float(g["trajectory_y_image"].iloc[-1]),
            "min_x_roi": float(g["trajectory_x_roi"].min()),
            "max_x_roi": float(g["trajectory_x_roi"].max()),
            "min_y_roi": float(g["trajectory_y_roi"].min()),
            "max_y_roi": float(g["trajectory_y_roi"].max()),
            "path_length_px_roi": float(path_len_px),
        }
        if "distance_m" in g.columns:
            row["min_distance_m"] = pd.to_numeric(g["distance_m"], errors="coerce").min()
        if "status" in g.columns:
            row["status_list"] = "|".join(sorted(set(str(x) for x in g["status"].dropna().unique())))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)


def build_trajectory_tables(
    csv_path: Path,
    root: Path,
    roi_path_arg: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[np.ndarray], str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV：{csv_path}")

    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"输入 CSV 为空：{csv_path}")

    vehicle_id_col = choose_column(raw, ["vehicle_id", "track_id", "id"])
    frame_col = choose_column(raw, ["frame_id", "frame", "frame_index"])
    time_col = choose_column(raw, ["time_s", "timestamp_s", "time"])
    class_col = choose_column(raw, ["vehicle_class", "class_name", "class", "vehicle_type"])

    if vehicle_id_col is None:
        raise ValueError("CSV 缺少 vehicle_id / track_id / id 列，无法区分每辆车。")
    if frame_col is None:
        raise ValueError("CSV 缺少 frame_id / frame 列，无法按时间排序轨迹。")

    x_col = choose_column(raw, [
        "trajectory_x",
        "vehicle_road_center_x",
        "vehicle_center_x",
        "bbox_center_x",
        "center_x",
        "x",
    ])
    y_col = choose_column(raw, [
        "trajectory_y",
        "ref_y",
        "road_profile_y",
        "vehicle_center_y",
        "bbox_center_y",
        "center_y",
        "y",
    ])

    if x_col is None or y_col is None:
        raise ValueError(
            "CSV 中找不到可用轨迹坐标列。需要 vehicle_road_center_x/vehicle_center_x/bbox_center_x "
            "和 ref_y/road_profile_y/vehicle_center_y/bbox_center_y。"
        )

    df = raw.copy()
    df[vehicle_id_col] = pd.to_numeric(df[vehicle_id_col], errors="coerce")
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[vehicle_id_col, frame_col, x_col, y_col]).copy()
    df[vehicle_id_col] = df[vehicle_id_col].astype(int)
    df[frame_col] = df[frame_col].astype(int)

    if EXPORT_ONLY_VALID_MEASUREMENT and "measurement_valid" in df.columns:
        df = df[to_bool_series(df["measurement_valid"])].copy()

    image_width, image_height, fps, size_source = infer_image_size(df, x_col, y_col, root)
    roi_polygon, roi_source = read_roi_config(root, roi_path_arg)
    origin_x, origin_y, origin_source, origin_selected_pts = compute_roi_top_left_origin(
        roi_polygon, ROI_ORIGIN_MODE
    )
    roi_polygon_relative = convert_roi_polygon_to_relative(roi_polygon, origin_x, origin_y)

    points_all = pd.DataFrame()
    points_all["vehicle_id"] = df[vehicle_id_col].astype(int)
    points_all["frame_id"] = df[frame_col].astype(int)

    if time_col and time_col in df.columns:
        points_all["time_s"] = pd.to_numeric(df[time_col], errors="coerce")
    elif fps and fps > 0:
        points_all["time_s"] = points_all["frame_id"] / float(fps)
    else:
        points_all["time_s"] = np.nan

    if class_col and class_col in df.columns:
        points_all["vehicle_class"] = df[class_col].astype(str)
    else:
        points_all["vehicle_class"] = "vehicle"

    points_all["trajectory_x_image"] = df[x_col].astype(float)
    points_all["trajectory_y_image"] = df[y_col].astype(float)
    points_all["roi_origin_x_image"] = float(origin_x)
    points_all["roi_origin_y_image"] = float(origin_y)

    x_roi, y_roi = convert_to_roi_xy(
        points_all["trajectory_x_image"],
        points_all["trajectory_y_image"],
        origin_x,
        origin_y,
    )
    points_all["trajectory_x_roi"] = x_roi
    points_all["trajectory_y_roi"] = y_roi

    # 兼容旧字段名。
    points_all["trajectory_x"] = points_all["trajectory_x_roi"]
    points_all["trajectory_y_xy"] = points_all["trajectory_y_roi"]

    # 关键：用 ROI polygon 判断点是否真实在 ROI 内。
    if roi_polygon is not None:
        points_all["inside_roi"] = [
            point_inside_roi_polygon(x, y, roi_polygon)
            for x, y in zip(points_all["trajectory_x_image"], points_all["trajectory_y_image"])
        ]
    else:
        # 没有 ROI 时，不能过滤；全部视为可用，并在说明里提示。
        points_all["inside_roi"] = True

    points_all["roi_origin_source"] = origin_source
    points_all["roi_origin_mode"] = ROI_ORIGIN_MODE

    optional_cols = [
        "status", "status_raw", "measurement_valid", "measurement_invalid_reason",
        "distance_m", "distance_m_smooth", "distance_px", "lane_left_id", "lane_right_id",
        "left_lane_x", "right_lane_x", "lane_width_px", "meter_per_pixel",
        "traffic_direction", "measured_visual_side", "measure_lane_x", "vehicle_measure_x",
        "bbox_center_x", "vehicle_center_x", "vehicle_road_center_x", "ref_y", "road_profile_y",
        "vehicle_left_x", "vehicle_right_x", "vehicle_type_params",
    ]
    for c in optional_cols:
        if c in df.columns and c not in points_all.columns:
            points_all[c] = df[c].values

    points_all = points_all.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)
    points_all = points_all.drop_duplicates(subset=["vehicle_id", "frame_id"], keep="last").reset_index(drop=True)
    points_all["point_index_all"] = points_all.groupby("vehicle_id").cumcount() + 1

    if FILTER_MAIN_OUTPUT_TO_ROI_INSIDE:
        points_main = points_all[points_all["inside_roi"].astype(bool)].copy()
    else:
        points_main = points_all.copy()

    points_main = points_main.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)
    points_main["point_index"] = points_main.groupby("vehicle_id").cumcount() + 1

    summary = build_summary(points_main)

    selected_points_text = ""
    if origin_selected_pts is not None:
        selected_points_text = json.dumps(origin_selected_pts.tolist(), ensure_ascii=False)

    meta = pd.DataFrame([
        {"key": "input_csv", "value": str(csv_path)},
        {"key": "roi_config", "value": roi_source},
        {"key": "x_source_column", "value": x_col},
        {"key": "y_source_column", "value": y_col},
        {"key": "image_width", "value": image_width},
        {"key": "image_height", "value": image_height},
        {"key": "image_size_source", "value": size_source},
        {"key": "roi_origin_x_image", "value": origin_x},
        {"key": "roi_origin_y_image", "value": origin_y},
        {"key": "roi_origin_source", "value": origin_source},
        {"key": "roi_origin_mode", "value": ROI_ORIGIN_MODE},
        {"key": "roi_origin_selected_points", "value": selected_points_text},
        {"key": "filter_main_output_to_roi_inside", "value": FILTER_MAIN_OUTPUT_TO_ROI_INSIDE},
        {"key": "all_points_count", "value": len(points_all)},
        {"key": "roi_inside_points_count", "value": int(points_all["inside_roi"].astype(bool).sum())},
        {"key": "roi_outside_points_count", "value": int((~points_all["inside_roi"].astype(bool)).sum())},
        {"key": "coordinate_note", "value": "以 ROI 外接矩形左上角为原点；X 正方向为图像向右；Y 正方向为图像向下；主表和主图只使用 inside_roi=True 的点；不做坐标裁剪。"},
        {"key": "export_only_valid_measurement", "value": EXPORT_ONLY_VALID_MEASUREMENT},
    ])

    return points_main, points_all, summary, meta, roi_polygon_relative, x_col, y_col


# =========================
# 绘图
# =========================

def _set_roi_axis_limits(ax, points: pd.DataFrame, roi_rel: Optional[np.ndarray]) -> None:
    xs = points["trajectory_x_roi"].to_numpy(dtype=float) if not points.empty else np.array([], dtype=float)
    ys = points["trajectory_y_roi"].to_numpy(dtype=float) if not points.empty else np.array([], dtype=float)

    if roi_rel is not None:
        xs = np.concatenate([xs, roi_rel[:, 0]])
        ys = np.concatenate([ys, roi_rel[:, 1]])

    if len(xs) == 0 or len(ys) == 0:
        return

    x_min, x_max = np.nanmin(xs), np.nanmax(xs)
    y_min, y_max = np.nanmin(ys), np.nanmax(ys)
    x_margin = max(30.0, 0.08 * max(1.0, x_max - x_min))
    y_margin = max(30.0, 0.08 * max(1.0, y_max - y_min))

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)


def draw_roi_relative(ax, roi_rel: Optional[np.ndarray]) -> None:
    if roi_rel is not None and len(roi_rel) >= 3:
        closed = np.vstack([roi_rel, roi_rel[0]])
        ax.plot(closed[:, 0], closed[:, 1], linestyle="--", linewidth=1.2, alpha=0.65, label="ROI boundary")
    ax.scatter([0], [0], s=45, marker="+", label="ROI bbox top-left origin (0,0)")
    ax.text(0, 0, "  XY(0,0)", fontsize=9, va="bottom")


def plot_overview_roi_xy(points: pd.DataFrame, roi_rel: Optional[np.ndarray], out_png: Path) -> None:
    ensure_dir(out_png.parent)
    fig, ax = plt.subplots(figsize=OVERVIEW_FIGSIZE)

    draw_roi_relative(ax, roi_rel)

    ids = sorted(points["vehicle_id"].unique()) if not points.empty else []
    for vid in ids:
        g = points[points["vehicle_id"] == vid].sort_values("frame_id")
        if len(g) < 2:
            ax.scatter(g["trajectory_x_roi"], g["trajectory_y_roi"], s=POINT_SIZE, label=f"ID{int(vid)}")
        else:
            ax.plot(
                g["trajectory_x_roi"],
                g["trajectory_y_roi"],
                marker="o",
                markersize=2.5,
                linewidth=LINE_WIDTH,
                label=f"ID{int(vid)}",
            )
        last = g.iloc[-1]
        ax.text(last["trajectory_x_roi"], last["trajectory_y_roi"], f"ID{int(vid)}", fontsize=8)

    ax.axhline(0, linewidth=0.8, alpha=0.45)
    ax.axvline(0, linewidth=0.8, alpha=0.45)
    ax.set_title("Vehicle Trajectories Inside ROI with ROI Bounding-box Top-left as XY Origin")
    ax.set_xlabel("X from ROI bbox top-left / pixel, positive to image right")
    ax.set_ylabel("Y from ROI bbox top-left / pixel, positive to image down")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
    _set_roi_axis_limits(ax, points, roi_rel)

    if len(ids) <= 30:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_each_vehicle(points: pd.DataFrame, roi_rel: Optional[np.ndarray], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for vid, g in points.groupby("vehicle_id", sort=True):
        g = g.sort_values("frame_id")
        if len(g) < MIN_POINTS_FOR_SINGLE_ID_PLOT:
            continue
        cls = str(g["vehicle_class"].mode().iloc[0]) if not g["vehicle_class"].dropna().empty else "vehicle"

        fig, ax = plt.subplots(figsize=SINGLE_ID_FIGSIZE)
        draw_roi_relative(ax, roi_rel)
        ax.plot(
            g["trajectory_x_roi"],
            g["trajectory_y_roi"],
            marker="o",
            markersize=4,
            linewidth=SINGLE_ID_LINE_WIDTH,
        )
        ax.scatter(g["trajectory_x_roi"].iloc[0], g["trajectory_y_roi"].iloc[0], s=SINGLE_ID_POINT_SIZE * 2, marker="o", label="ROI-inside track start")
        ax.scatter(g["trajectory_x_roi"].iloc[-1], g["trajectory_y_roi"].iloc[-1], s=SINGLE_ID_POINT_SIZE * 2, marker="x", label="ROI-inside track end")
        ax.axhline(0, linewidth=0.8, alpha=0.45)
        ax.axvline(0, linewidth=0.8, alpha=0.45)
        ax.set_title(f"ID{int(vid)} ROI-inside Trajectory ({cls}), frames {int(g['frame_id'].min())}-{int(g['frame_id'].max())}")
        ax.set_xlabel("X from ROI bbox top-left / pixel")
        ax.set_ylabel("Y from ROI bbox top-left / pixel")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.55)
        _set_roi_axis_limits(ax, g, roi_rel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"ID{int(vid):03d}_{cls}_roi_bbox_topleft_xy.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


# =========================
# Excel 输出
# =========================

def safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\\/*?:\[\]]", "_", str(name))
    return name[:31]


def autofit_columns_xlsxwriter(worksheet, df: pd.DataFrame, max_width: int = 32) -> None:
    for idx, col in enumerate(df.columns):
        values = df[col].astype(str).head(500)
        width = max([len(str(col))] + [len(v) for v in values]) + 2
        worksheet.set_column(idx, idx, min(max(width, 8), max_width))


def write_excel(
    out_xlsx: Path,
    points_main: pd.DataFrame,
    points_all: pd.DataFrame,
    summary: pd.DataFrame,
    meta: pd.DataFrame,
    overview_png: Path,
) -> None:
    ensure_dir(out_xlsx.parent)

    try:
        writer = pd.ExcelWriter(out_xlsx, engine="xlsxwriter")
        engine = "xlsxwriter"
    except Exception:
        writer = pd.ExcelWriter(out_xlsx)
        engine = "fallback"

    with writer:
        meta.to_excel(writer, sheet_name="说明", index=False)
        summary.to_excel(writer, sheet_name="车辆轨迹汇总", index=False)
        points_main.to_excel(writer, sheet_name="ROI内轨迹坐标", index=False)
        points_all.to_excel(writer, sheet_name="原始全部轨迹坐标", index=False)

        if WRITE_EACH_VEHICLE_SHEET:
            unique_ids = sorted(points_main["vehicle_id"].unique()) if not points_main.empty else []
            if len(unique_ids) <= MAX_VEHICLE_SHEETS:
                for vid in unique_ids:
                    g = points_main[points_main["vehicle_id"] == vid].sort_values("frame_id")
                    cls = str(g["vehicle_class"].mode().iloc[0]) if not g["vehicle_class"].dropna().empty else "vehicle"
                    sheet = safe_sheet_name(f"ID{int(vid):03d}_{cls}")
                    g.to_excel(writer, sheet_name=sheet, index=False)

        if engine == "xlsxwriter":
            workbook = writer.book
            header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
            num_fmt = workbook.add_format({"num_format": "0.000"})
            int_fmt = workbook.add_format({"num_format": "0"})

            sheet_map = [
                ("说明", meta),
                ("车辆轨迹汇总", summary),
                ("ROI内轨迹坐标", points_main),
                ("原始全部轨迹坐标", points_all),
            ]
            for sheet_name, df in sheet_map:
                ws = writer.sheets[sheet_name]
                ws.freeze_panes(1, 0)
                if not df.empty:
                    ws.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
                for c_idx, col in enumerate(df.columns):
                    ws.write(0, c_idx, col, header_fmt)
                autofit_columns_xlsxwriter(ws, df)

            if WRITE_EACH_VEHICLE_SHEET and len(points_main["vehicle_id"].unique()) <= MAX_VEHICLE_SHEETS:
                for sheet_name, ws in writer.sheets.items():
                    if sheet_name.startswith("ID"):
                        ws.freeze_panes(1, 0)

            chart_ws = workbook.add_worksheet("轨迹图")
            chart_ws.write("A1", "ROI 内车辆 XY 轨迹图", header_fmt)
            chart_ws.write("A2", "坐标说明：以 ROI 外接矩形左上角为原点；X 轴正方向为图像向右；Y 轴正方向为图像向下；主图只绘制 inside_roi=True 的轨迹点。")
            chart_ws.write("A3", "说明：这不是坐标裁剪；ROI 外点已保存在“原始全部轨迹坐标”sheet 中，不参与主图和主汇总。")
            if overview_png.exists():
                chart_ws.insert_image("A5", str(overview_png), {"x_scale": 0.82, "y_scale": 0.82})

            for sheet_name, df in [
                ("车辆轨迹汇总", summary),
                ("ROI内轨迹坐标", points_main),
                ("原始全部轨迹坐标", points_all),
            ]:
                ws = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    if col in {"vehicle_id", "frame_id", "point_index", "point_index_all", "start_frame", "end_frame", "roi_inside_point_count"}:
                        ws.set_column(idx, idx, 10, int_fmt)
                    elif any(k in col for k in ["x", "y", "distance", "duration", "length", "ratio"]):
                        ws.set_column(idx, idx, 14, num_fmt)


# =========================
# 主入口
# =========================

def parse_args() -> argparse.Namespace:
    root = get_project_root()
    default_csv = root / "outputs" / "vehicle_distance.csv"
    default_xlsx = root / "outputs" / "vehicle_trajectory_roi_bbox_topleft_xy.xlsx"
    default_overview = root / "outputs" / "vehicle_trajectory_roi_bbox_topleft_xy_overview.png"
    default_by_id_dir = root / "outputs" / "vehicle_trajectory_roi_bbox_topleft_by_id"

    parser = argparse.ArgumentParser(description="导出 ROI 内部车辆轨迹坐标 Excel 和 XY 图，以 ROI 外接矩形左上角为原点。")
    parser.add_argument("--csv", type=str, default=str(default_csv), help="输入 vehicle_distance.csv 路径")
    parser.add_argument("--roi", type=str, default=None, help="ROI 配置路径，默认 configs/roi_config.json")
    parser.add_argument("--out-xlsx", type=str, default=str(default_xlsx), help="输出 Excel 路径")
    parser.add_argument("--overview", type=str, default=str(default_overview), help="输出 XY 坐标总览图")
    parser.add_argument("--by-id-dir", type=str, default=str(default_by_id_dir), help="每辆车单独轨迹图输出目录")
    parser.add_argument("--origin-mode", type=str, default=ROI_ORIGIN_MODE, choices=["BOUNDING_TOP_LEFT", "VERTEX_TOP_LEFT"], help="ROI 原点模式")
    parser.add_argument("--keep-all-points", action="store_true", help="主表和主图不按 inside_roi 过滤。默认过滤，只保留 ROI 内点。")
    return parser.parse_args()


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else root / path


def main() -> None:
    global ROI_ORIGIN_MODE, FILTER_MAIN_OUTPUT_TO_ROI_INSIDE

    root = get_project_root()
    args = parse_args()
    ROI_ORIGIN_MODE = args.origin_mode
    if args.keep_all_points:
        FILTER_MAIN_OUTPUT_TO_ROI_INSIDE = False

    csv_path = resolve_path(root, args.csv)
    out_xlsx = resolve_path(root, args.out_xlsx)
    overview = resolve_path(root, args.overview)
    by_id_dir = resolve_path(root, args.by_id_dir)

    points_main, points_all, summary, meta, roi_rel, x_col, y_col = build_trajectory_tables(
        csv_path=csv_path,
        root=root,
        roi_path_arg=args.roi,
    )

    if points_all.empty:
        raise ValueError("没有可用轨迹点，无法导出。")
    if points_main.empty:
        raise ValueError("按 ROI polygon 过滤后没有轨迹点。请检查 ROI 配置或使用 --keep-all-points 调试。")

    plot_overview_roi_xy(points_main, roi_rel, overview)
    plot_each_vehicle(points_main, roi_rel, by_id_dir)
    write_excel(out_xlsx, points_main, points_all, summary, meta, overview)

    print("[OK] ROI 内轨迹 Excel 已输出：", out_xlsx)
    print("[OK] ROI 内轨迹总览图已输出：", overview)
    print("[OK] 每辆车 ROI 内单独轨迹图目录：", by_id_dir)
    print(f"[INFO] 轨迹坐标来源：x={x_col}, y={y_col}")
    print(f"[INFO] 原始轨迹点数：{points_all.shape[0]}，ROI 内轨迹点数：{points_main.shape[0]}，车辆数：{summary.shape[0]}")
    if not meta.empty:
        md = dict(zip(meta["key"], meta["value"]))
        print(f"[INFO] ROI 外接矩形左上角原点：x={md.get('roi_origin_x_image')}, y={md.get('roi_origin_y_image')}")
        print(f"[INFO] ROI 原点模式：{md.get('roi_origin_mode')}，来源：{md.get('roi_origin_source')}")


if __name__ == "__main__":
    main()
