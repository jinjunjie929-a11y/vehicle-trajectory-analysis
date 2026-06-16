from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml_config() -> dict[str, Any]:
    """Load config.yaml when PyYAML is available; fall back to defaults otherwise."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {CONFIG_PATH}")
    return data


CONFIG = _load_yaml_config()


def _get(path: str, default: Any) -> Any:
    current: Any = CONFIG
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _path_name(path: str, default: str) -> str:
    value = str(_get(path, default))
    return Path(value).name if value else default


# =========================
# Project inputs
# =========================
VIDEO_NAME = _path_name("paths.video", "highway_03.mp4")
MODEL_NAME = _path_name("paths.model", "yolov8n-seg.pt")
ROI_CONFIG_NAME = _path_name("paths.roi", "roi_config.json")


# =========================
# Project outputs
# =========================
LANE_MAP_NAME = _path_name("outputs.lane_map", "stable_lane_map.json")
RIGHT_DISTANCE_VIDEO_NAME = _path_name("outputs.distance_video", "vehicle_distance.mp4")
RIGHT_DISTANCE_CSV_NAME = _path_name("outputs.distance_csv", "vehicle_distance.csv")

TRAJECTORY_XLSX_NAME = _path_name("outputs.trajectory_excel", "vehicle_trajectory_roi_xy.xlsx")
TRAJECTORY_OVERVIEW_NAME = _path_name("outputs.trajectory_overview", "vehicle_trajectory_roi_xy_overview.png")
TRAJECTORY_BY_ID_DIR_NAME = _path_name("outputs.trajectory_by_id_dir", "vehicle_trajectory_roi_xy_by_id")
TRAJECTORY_RENDER_DIR_NAME = _path_name("outputs.trajectory_render_dir", "vehicle_trajectory_render")


# =========================
# Video and lane settings
# =========================
BASE_WIDTH = int(_get("video.base_width", 1280))
BASE_HEIGHT = int(_get("video.base_height", 720))
BASE_FPS = float(_get("video.base_fps", 50.0))

START_FRAME = int(_get("video.start_frame", 0))
PROCESS_SECONDS = _get("video.process_seconds", 30.0)
if PROCESS_SECONDS is not None:
    PROCESS_SECONDS = float(PROCESS_SECONDS)

LANE_MAP_SAMPLE_SECONDS = _get("lane_map.sample_seconds", 30.0)
if LANE_MAP_SAMPLE_SECONDS is not None:
    LANE_MAP_SAMPLE_SECONDS = float(LANE_MAP_SAMPLE_SECONDS)
LANE_MAP_SAMPLE_INTERVAL_SECONDS = float(_get("lane_map.sample_interval_seconds", 0.5))
TARGET_LANE_COUNT = int(_get("lane_map.target_lane_count", 5))
MIN_VALID_LANE_MAP_FRAMES = int(_get("lane_map.min_valid_frames", 8))

TRAFFIC_DIRECTION = str(_get("traffic.direction", "TOWARD_CAMERA")).upper()
LANE_WIDTH_M = float(_get("traffic.lane_width_m", 3.75))


# =========================
# Detection settings
# =========================
YOLO_CONF = float(_get("detection.yolo_conf", 0.08))
YOLO_IOU = float(_get("detection.yolo_iou", 0.45))
CAR_MIN_CONF = float(_get("detection.class_confidence.car", 0.18))
TRUCK_MIN_CONF = float(_get("detection.class_confidence.truck", 0.18))
BUS_MIN_CONF = float(_get("detection.class_confidence.bus", 0.18))
MOTORCYCLE_MIN_CONF = float(_get("detection.class_confidence.motorcycle", 0.08))


def project_root_from_file(file_path: str | Path) -> Path:
    """Return the repository root for scripts stored in src/."""
    p = Path(file_path).resolve()
    if p.parent.name.lower() == "src":
        return p.parents[1]
    return Path.cwd().resolve()


def _resolve(root_dir: Path, configured_path: str, default_folder: str, default_name: str) -> Path:
    value = str(_get(configured_path, ""))
    if value:
        path = Path(value)
        return path if path.is_absolute() else root_dir / path
    return root_dir / default_folder / default_name


def video_path(root_dir: Path) -> Path:
    return _resolve(root_dir, "paths.video", "data", VIDEO_NAME)


def model_path(root_dir: Path) -> Path:
    return _resolve(root_dir, "paths.model", "models", MODEL_NAME)


def roi_config_path(root_dir: Path) -> Path:
    return _resolve(root_dir, "paths.roi", "configs", ROI_CONFIG_NAME)


def lane_map_path(root_dir: Path) -> Path:
    return _resolve(root_dir, "outputs.lane_map", "outputs", LANE_MAP_NAME)


def outputs_dir(root_dir: Path) -> Path:
    return root_dir / "outputs"


def debug_dir(root_dir: Path) -> Path:
    return root_dir / "debug"


def configs_dir(root_dir: Path) -> Path:
    return root_dir / "configs"


def ensure_project_dirs(root_dir: Path) -> None:
    for name in ["data", "models", "configs", "debug", "outputs"]:
        (root_dir / name).mkdir(exist_ok=True)


def read_video_meta(video_file: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_file}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 1e-6:
        fps = BASE_FPS

    duration_sec = frame_count / fps if fps > 0 else 0.0
    return {
        "video_path": str(video_file),
        "video_name": video_file.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def write_video_meta(root_dir: Path, meta: dict[str, Any]) -> Path:
    out = outputs_dir(root_dir) / "video_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out


def frames_from_seconds(seconds: float, fps: float, min_frames: int = 1) -> int:
    if fps <= 1e-6:
        fps = BASE_FPS
    return max(min_frames, int(round(float(seconds) * float(fps))))


def process_frame_count(fps: float, frame_count: int, start_frame: int = START_FRAME) -> int:
    remaining = max(0, int(frame_count) - int(start_frame))
    if PROCESS_SECONDS is None:
        return remaining
    return min(remaining, frames_from_seconds(PROCESS_SECONDS, fps, min_frames=1))


def lane_map_sample_end_frame(fps: float, frame_count: int, start_frame: int = START_FRAME) -> int:
    if LANE_MAP_SAMPLE_SECONDS is None:
        return max(0, int(frame_count) - 1)
    n = frames_from_seconds(LANE_MAP_SAMPLE_SECONDS, fps, min_frames=1)
    return min(max(0, int(frame_count) - 1), int(start_frame) + n)


def lane_map_sample_step_frames(fps: float) -> int:
    return frames_from_seconds(LANE_MAP_SAMPLE_INTERVAL_SECONDS, fps, min_frames=1)


def image_scale(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 1.0
    cur_diag = math.hypot(float(width), float(height))
    base_diag = math.hypot(float(BASE_WIDTH), float(BASE_HEIGHT))
    return max(0.35, min(3.0, cur_diag / base_diag))


def scale_px(value: float, width: int, height: int, min_value: int = 1) -> int:
    return max(min_value, int(round(float(value) * image_scale(width, height))))


def scaled_odd(value: float, width: int, height: int, min_value: int = 3) -> int:
    v = scale_px(value, width, height, min_value=min_value)
    if v % 2 == 0:
        v += 1
    return v


def auto_yolo_imgsz(width: int, height: int) -> int:
    max_side = max(int(width), int(height))
    target = int(round(max_side / 32.0) * 32)
    return max(640, min(1536, target))


def lane_group_scaled_params(width: int, height: int) -> dict[str, int]:
    return {
        "hough_threshold": scale_px(25, width, height, min_value=12),
        "hough_min_line_length": scale_px(25, width, height, min_value=12),
        "hough_max_line_gap": scale_px(90, width, height, min_value=30),
        "cluster_gap_px": scale_px(85, width, height, min_value=30),
        "point_distance_px": scale_px(14, width, height, min_value=6),
        "min_points_per_lane": max(30, int(round(80 * image_scale(width, height)))),
        "merge_duplicate_gap_px": scale_px(45, width, height, min_value=18),
        "bottom_ignore_px": scale_px(25, width, height, min_value=8),
        "residual_min_px": scale_px(18, width, height, min_value=8),
        "paint_patch_x": scale_px(8, width, height, min_value=3),
        "paint_patch_y": scale_px(5, width, height, min_value=2),
        "paint_min_pixels": max(2, int(round(3 * image_scale(width, height)))),
        "draw_thickness": scale_px(3, width, height, min_value=1),
    }
