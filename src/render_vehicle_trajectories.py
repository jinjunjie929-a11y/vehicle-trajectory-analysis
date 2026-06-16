# -*- coding: utf-8 -*-
r"""
render_vehicle_trajectories.py

功能：
1. 读取 outputs/vehicle_distance.csv 中每辆车的逐帧位置；
2. 为每辆车生成轨迹曲线；
3. 输出：
   - outputs/vehicle_trajectory_overlay.mp4       带所有车辆轨迹曲线的视频
   - outputs/vehicle_trajectory_overview.png      所有车辆总轨迹图
   - outputs/vehicle_trajectory_by_id/IDxxx_*.png 每辆车单独轨迹图
   - outputs/vehicle_trajectory_summary.csv       每辆车轨迹统计表

使用方式：
    cd D:\zhuanli
    python src\render_vehicle_trajectories.py

也可以指定输入输出：
    python src\render_vehicle_trajectories.py --csv outputs\vehicle_distance.csv --video data\highway_03.mp4

说明：
- 轨迹点默认使用 vehicle_road_center_x + ref_y，表示车辆在道路上的参考位置。
- 如果没有 vehicle_road_center_x，则退回 vehicle_center_x / bbox_center_x。
- 如果没有 ref_y，则退回 road_profile_y。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# =========================
# 可视化参数
# =========================
DRAW_ONLY_MEASUREMENT_VALID = False   # True：只画 measurement_valid=1 的点；False：有位置就画
MIN_TRACK_POINTS = 2                  # 少于该点数的 ID 不输出单独轨迹图
MAX_HISTORY_POINTS = 180              # 视频里每辆车最多显示最近多少个历史点；None 表示显示完整历史
TRAJECTORY_THICKNESS = 3
CURRENT_POINT_RADIUS = 5
START_POINT_RADIUS = 4
END_POINT_RADIUS = 5
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 2
OVERLAY_ALPHA = 0.92

# 摩托车轨迹/框统一红色；普通车辆按 ID 分配颜色
MOTORCYCLE_COLOR = (0, 0, 255)        # BGR 红色
TEXT_BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)

# 总览图是否使用首帧作为背景；False 则用黑底图
USE_FIRST_FRAME_AS_BACKGROUND = True


# =========================
# 路径与配置读取
# =========================

def get_project_root() -> Path:
    """兼容放在 src 下运行和在根目录运行。"""
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "src":
        return here.parent.parent
    return Path.cwd().resolve()


def try_import_project_config(root: Path):
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        import project_config as cfg  # type: ignore
        return cfg
    except Exception:
        return None


def default_video_path(root: Path, cfg) -> Optional[Path]:
    """尽量从 project_config.py 自动找到视频路径。"""
    if cfg is None:
        return None

    candidates: List[Path] = []

    for attr in ["VIDEO_PATH", "INPUT_VIDEO_PATH", "VIDEO_FILE"]:
        value = getattr(cfg, attr, None)
        if value:
            p = Path(str(value))
            candidates.append(p if p.is_absolute() else root / p)

    video_name = getattr(cfg, "VIDEO_NAME", None)
    if video_name:
        candidates.extend([
            root / "videos" / str(video_name),
            root / str(video_name),
            root / "data" / str(video_name),
        ])

    for p in candidates:
        if p.exists():
            return p

    return candidates[0] if candidates else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================
# 轨迹数据处理
# =========================

def choose_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float) != 0
    return s.astype(str).str.lower().isin(["1", "true", "yes", "ok", "valid"])


def load_tracks(csv_path: Path) -> Tuple[pd.DataFrame, str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到轨迹 CSV：{csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV 为空：{csv_path}")

    required = ["frame_id", "vehicle_id"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列：{col}")

    x_col = choose_column(df, [
        "trajectory_x",
        "vehicle_road_center_x",
        "vehicle_center_x",
        "bbox_center_x",
    ])
    y_col = choose_column(df, [
        "trajectory_y",
        "ref_y",
        "road_profile_y",
        "bbox_center_y",
        "vehicle_center_y",
    ])

    if x_col is None or y_col is None:
        raise ValueError(
            "CSV 中找不到可用轨迹坐标列。需要 vehicle_road_center_x / vehicle_center_x 和 ref_y / road_profile_y。"
        )

    df = df.copy()
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df["vehicle_id"] = pd.to_numeric(df["vehicle_id"], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    df = df.dropna(subset=["frame_id", "vehicle_id", x_col, y_col])
    df["frame_id"] = df["frame_id"].astype(int)
    df["vehicle_id"] = df["vehicle_id"].astype(int)

    if DRAW_ONLY_MEASUREMENT_VALID and "measurement_valid" in df.columns:
        df = df[to_bool_series(df["measurement_valid"])]

    if "vehicle_class" not in df.columns:
        df["vehicle_class"] = "vehicle"

    df = df.sort_values(["frame_id", "vehicle_id"]).reset_index(drop=True)
    return df, x_col, y_col


def stable_color_for_id(vehicle_id: int, vehicle_class: str = "") -> Tuple[int, int, int]:
    """给每个 ID 一个稳定且醒目的 BGR 颜色；摩托车固定红色。"""
    if str(vehicle_class).lower() == "motorcycle":
        return MOTORCYCLE_COLOR

    # 预设高对比颜色，避开太暗颜色
    palette = [
        (255, 255, 0),    # cyan
        (255, 0, 255),    # magenta
        (0, 255, 255),    # yellow
        (0, 255, 0),      # green
        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
        (255, 0, 128),
        (128, 0, 255),
        (0, 255, 128),
    ]
    return palette[int(vehicle_id) % len(palette)]


def draw_label(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, LABEL_FONT_SCALE, LABEL_THICKNESS)
    x = max(0, min(img.shape[1] - tw - 4, x))
    y = max(th + 4, min(img.shape[0] - 4, y))
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), TEXT_BG_COLOR, -1)
    cv2.putText(img, text, (x, y), font, LABEL_FONT_SCALE, color, LABEL_THICKNESS, cv2.LINE_AA)


def draw_polyline(
    img: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = TRAJECTORY_THICKNESS,
) -> None:
    if len(points) < 2:
        return
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def valid_point(x: float, y: float, w: int, h: int) -> bool:
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    return -100 <= x <= w + 100 and -100 <= y <= h + 100


def build_track_points(df: pd.DataFrame, x_col: str, y_col: str, w: int, h: int) -> Dict[int, List[Tuple[int, int, int, str]]]:
    """返回 {vehicle_id: [(frame_id, x, y, class_name), ...]}"""
    tracks: Dict[int, List[Tuple[int, int, int, str]]] = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        vid = int(d["vehicle_id"])
        frame_id = int(d["frame_id"])
        x = float(d[x_col])
        y = float(d[y_col])
        cls = str(d.get("vehicle_class", "vehicle"))
        if not valid_point(x, y, w, h):
            continue
        tracks.setdefault(vid, []).append((frame_id, int(round(x)), int(round(y)), cls))

    for vid in list(tracks.keys()):
        tracks[vid] = sorted(tracks[vid], key=lambda t: t[0])
    return tracks


def get_frame_size_and_fps(video_path: Optional[Path], df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[int, int, float, Optional[np.ndarray]]:
    default_w = int(max(1280, np.nanmax(pd.to_numeric(df[x_col], errors="coerce")) + 80))
    default_h = int(max(720, np.nanmax(pd.to_numeric(df[y_col], errors="coerce")) + 60))
    fps = 25.0
    first_frame = None

    if video_path is not None and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = float(cap.get(cv2.CAP_PROP_FPS))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                first_frame = frame
            if w > 0 and h > 0:
                default_w, default_h = w, h
            if f > 1:
                fps = f

    return default_w, default_h, fps, first_frame


# =========================
# 输出：总览图、每车单图、视频
# =========================

def draw_all_tracks_overview(
    tracks: Dict[int, List[Tuple[int, int, int, str]]],
    background: Optional[np.ndarray],
    output_path: Path,
    w: int,
    h: int,
) -> None:
    if background is not None and USE_FIRST_FRAME_AS_BACKGROUND:
        canvas = background.copy()
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    overlay = canvas.copy()
    for vid, pts_info in sorted(tracks.items()):
        if len(pts_info) < MIN_TRACK_POINTS:
            continue
        cls = pts_info[0][3]
        color = stable_color_for_id(vid, cls)
        pts = [(x, y) for _, x, y, _ in pts_info]
        draw_polyline(overlay, pts, color, TRAJECTORY_THICKNESS)
        cv2.circle(overlay, pts[0], START_POINT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, pts[-1], END_POINT_RADIUS, color, -1, cv2.LINE_AA)
        draw_label(overlay, f"ID{vid} {cls}", (pts[-1][0] + 6, pts[-1][1] - 6), color)

    canvas = cv2.addWeighted(overlay, OVERLAY_ALPHA, canvas, 1 - OVERLAY_ALPHA, 0)
    cv2.imwrite(str(output_path), canvas)


def draw_single_track_images(
    tracks: Dict[int, List[Tuple[int, int, int, str]]],
    background: Optional[np.ndarray],
    out_dir: Path,
    w: int,
    h: int,
) -> None:
    ensure_dir(out_dir)
    for vid, pts_info in sorted(tracks.items()):
        if len(pts_info) < MIN_TRACK_POINTS:
            continue
        cls = pts_info[0][3]
        color = stable_color_for_id(vid, cls)
        canvas = background.copy() if background is not None and USE_FIRST_FRAME_AS_BACKGROUND else np.zeros((h, w, 3), dtype=np.uint8)
        pts = [(x, y) for _, x, y, _ in pts_info]
        draw_polyline(canvas, pts, color, TRAJECTORY_THICKNESS + 1)
        cv2.circle(canvas, pts[0], START_POINT_RADIUS + 1, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], END_POINT_RADIUS + 1, color, -1, cv2.LINE_AA)
        draw_label(canvas, f"ID{vid} {cls}  frames {pts_info[0][0]}-{pts_info[-1][0]}", (20, 40), color)
        out_path = out_dir / f"ID{vid:03d}_{cls}_trajectory.png"
        cv2.imwrite(str(out_path), canvas)


def write_track_summary(
    tracks: Dict[int, List[Tuple[int, int, int, str]]],
    summary_path: Path,
) -> None:
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vehicle_id",
            "vehicle_class",
            "start_frame",
            "end_frame",
            "point_count",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "delta_x",
            "delta_y",
            "path_length_px",
        ])
        for vid, pts_info in sorted(tracks.items()):
            if len(pts_info) < MIN_TRACK_POINTS:
                continue
            cls = pts_info[0][3]
            pts = [(x, y) for _, x, y, _ in pts_info]
            path_len = 0.0
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                path_len += math.hypot(x2 - x1, y2 - y1)
            writer.writerow([
                vid,
                cls,
                pts_info[0][0],
                pts_info[-1][0],
                len(pts_info),
                pts[0][0],
                pts[0][1],
                pts[-1][0],
                pts[-1][1],
                pts[-1][0] - pts[0][0],
                pts[-1][1] - pts[0][1],
                round(path_len, 2),
            ])


def render_trajectory_video(
    video_path: Path,
    tracks: Dict[int, List[Tuple[int, int, int, str]]],
    output_path: Path,
    fps: float,
    w: int,
    h: int,
) -> None:
    if not video_path.exists():
        print(f"[WARN] 找不到视频，跳过轨迹视频输出：{video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频，跳过轨迹视频输出：{video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    # frame_id -> [(vid, x, y, cls)]
    points_by_frame: Dict[int, List[Tuple[int, int, int, str]]] = {}
    min_frame = None
    max_frame = None
    for vid, pts_info in tracks.items():
        for frame_id, x, y, cls in pts_info:
            points_by_frame.setdefault(frame_id, []).append((vid, x, y, cls))
            min_frame = frame_id if min_frame is None else min(min_frame, frame_id)
            max_frame = frame_id if max_frame is None else max(max_frame, frame_id)

    history: Dict[int, List[Tuple[int, int, str]]] = {}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        for vid, x, y, cls in points_by_frame.get(frame_idx, []):
            history.setdefault(vid, []).append((x, y, cls))
            if MAX_HISTORY_POINTS is not None and len(history[vid]) > MAX_HISTORY_POINTS:
                history[vid] = history[vid][-MAX_HISTORY_POINTS:]

        overlay = frame.copy()

        for vid, hist in sorted(history.items()):
            if len(hist) < 2:
                continue
            cls = hist[-1][2]
            color = stable_color_for_id(vid, cls)
            pts = [(x, y) for x, y, _ in hist]
            draw_polyline(overlay, pts, color, TRAJECTORY_THICKNESS)
            cv2.circle(overlay, pts[-1], CURRENT_POINT_RADIUS, color, -1, cv2.LINE_AA)
            draw_label(overlay, f"ID{vid}", (pts[-1][0] + 6, pts[-1][1] - 6), color)

        frame = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0)
        cv2.putText(
            frame,
            "vehicle trajectories",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

        # 如果 CSV 只处理前若干秒，可以在 max_frame 后多输出 10 帧后停止，避免把完整长视频都输出。
        if max_frame is not None and frame_idx > max_frame + 10:
            break

    cap.release()
    writer.release()


# =========================
# 主函数
# =========================

def parse_args() -> argparse.Namespace:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="输出每辆车的轨迹曲线图和轨迹视频")
    parser.add_argument("--csv", type=str, default=str(root / "outputs" / "vehicle_distance.csv"), help="逐帧测距 CSV 路径")
    parser.add_argument("--video", type=str, default="", help="原始视频路径；不填则尝试从 project_config.py 自动读取")
    parser.add_argument("--output-dir", type=str, default=str(root / "outputs"), help="输出目录")
    parser.add_argument("--no-video", action="store_true", help="只输出图片和 summary，不输出轨迹视频")
    parser.add_argument("--valid-only", action="store_true", help="只用 measurement_valid=1 的点绘制轨迹")
    return parser.parse_args()


def main() -> None:
    global DRAW_ONLY_MEASUREMENT_VALID

    args = parse_args()
    root = get_project_root()
    cfg = try_import_project_config(root)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    video_path: Optional[Path]
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = root / video_path
    else:
        video_path = default_video_path(root, cfg)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    ensure_dir(out_dir)

    DRAW_ONLY_MEASUREMENT_VALID = bool(args.valid_only)

    print(f"[INFO] CSV: {csv_path}")
    if video_path is not None:
        print(f"[INFO] Video: {video_path}")

    df, x_col, y_col = load_tracks(csv_path)
    w, h, fps, first_frame = get_frame_size_and_fps(video_path, df, x_col, y_col)
    tracks = build_track_points(df, x_col, y_col, w, h)

    valid_track_count = sum(1 for pts in tracks.values() if len(pts) >= MIN_TRACK_POINTS)
    print(f"[INFO] 坐标列: x={x_col}, y={y_col}")
    print(f"[INFO] 图像尺寸: {w}x{h}, fps={fps:.3f}")
    print(f"[INFO] 轨迹车辆数: {valid_track_count}")

    overview_path = out_dir / "vehicle_trajectory_overview.png"
    by_id_dir = out_dir / "vehicle_trajectory_by_id"
    summary_path = out_dir / "vehicle_trajectory_summary.csv"
    video_out_path = out_dir / "vehicle_trajectory_overlay.mp4"

    draw_all_tracks_overview(tracks, first_frame, overview_path, w, h)
    draw_single_track_images(tracks, first_frame, by_id_dir, w, h)
    write_track_summary(tracks, summary_path)

    if not args.no_video and video_path is not None:
        render_trajectory_video(video_path, tracks, video_out_path, fps, w, h)

    print("[OK] 轨迹输出完成：")
    print(f"  总览图: {overview_path}")
    print(f"  单车轨迹目录: {by_id_dir}")
    print(f"  轨迹统计表: {summary_path}")
    if not args.no_video:
        print(f"  轨迹视频: {video_out_path}")


if __name__ == "__main__":
    main()
