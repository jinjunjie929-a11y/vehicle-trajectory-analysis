from pathlib import Path
import json
from collections import Counter

import cv2
import numpy as np

import project_config as cfg

from lane_group_fit import (
    load_roi_mask,
    extract_lane_candidates,
    detect_hough_line_models,
    cluster_hough_models,
    merge_cluster_to_line_model,
    collect_points_near_line,
    fit_quadratic_x_by_y,
    extract_paint_segments,
    sample_polyline,
    merge_duplicate_lanes,
)


# =========================
# 可调参数
# =========================

VIDEO_NAME = cfg.VIDEO_NAME

# 先用前 1500 帧建稳定车道地图。
# 你的 FPS 是 50，1500 帧约等于 30 秒。
SAMPLE_START_FRAME = 0
SAMPLE_END_FRAME = 1500
SAMPLE_STEP = 25          # 每隔 25 帧取一帧，相当于 0.5 秒取一帧

# 从 project_config.py 读取目标车道边界线数量。
# 后续换视频时只需要改 cfg.TARGET_LANE_COUNT，不要在本文件里写死。
TARGET_LANE_COUNT = int(cfg.TARGET_LANE_COUNT)

# 至少需要多少个有效采样帧才能生成稳定地图
MIN_VALID_FRAMES = int(getattr(cfg, "MIN_VALID_LANE_MAP_FRAMES", 8))

# 稳定候选图投票阈值。
# 数值越大，越只保留长期稳定出现的白线；
# 数值越小，越容易保留噪声。
VOTE_RATIO = 0.08

DRAW_CONTINUOUS_THICKNESS = 3
DRAW_PAINT_THICKNESS = 5


def fit_lanes_in_frame(frame, roi_mask, roi_points):
    """
    对单帧执行：
    白线候选提取 → Hough 粗方向 → 聚类 → 二次拟合 → 得到 lane 0~lane N。
    """
    y_top = int(np.min(roi_points[:, 1]))
    y_bottom = int(np.max(roi_points[:, 1]) - cfg.scale_px(25, frame.shape[1], frame.shape[0], min_value=8))
    y_ref = y_bottom

    candidate_mask = extract_lane_candidates(frame, roi_mask)

    hough_models = detect_hough_line_models(candidate_mask)
    clusters = cluster_hough_models(hough_models, y_ref)
    rough_models = [merge_cluster_to_line_model(cluster) for cluster in clusters]

    lanes = []

    for rough_id, rough_model in enumerate(rough_models):
        pts = collect_points_near_line(candidate_mask, rough_model, roi_mask)
        fit_result = fit_quadratic_x_by_y(pts)

        if fit_result is None:
            continue

        coeff, good_pts = fit_result

        y_min = max(y_top, float(np.min(good_pts[:, 1])))
        y_max = min(y_bottom, float(np.max(good_pts[:, 1])))

        # 过短的线不作为稳定车道线
        if y_max - y_min < 80:
            continue

        paint_segments, coverage, lane_type = extract_paint_segments(
            coeff,
            candidate_mask,
            roi_mask,
            y_min,
            y_max,
            step=5,
        )

        lanes.append(
            {
                "rough_id": rough_id,
                "coeff": coeff,
                "point_count": int(len(good_pts)),
                "y_min": float(y_min),
                "y_max": float(y_max),
                "paint_segments_y": paint_segments,
                "paint_coverage": float(coverage),
                "lane_type": lane_type,
            }
        )

    lanes = merge_duplicate_lanes(lanes, y_ref)

    # 按 y_ref 处的 x 坐标从左到右排序
    for lane in lanes:
        lane["x_ref"] = float(np.polyval(lane["coeff"], y_ref))

    lanes = sorted(lanes, key=lambda item: item["x_ref"])

    for i, lane in enumerate(lanes):
        lane["lane_id"] = i

    return candidate_mask, lanes, {
        "y_top": y_top,
        "y_bottom": y_bottom,
        "y_ref": y_ref,
        "num_hough_models": len(hough_models),
        "num_hough_clusters": len(clusters),
    }


def aggregate_lanes(valid_records, roi_mask, display_candidate_mask, y_top, y_bottom, y_ref):
    """
    对多帧 lane 0~lane N 的二次拟合参数取中位数，
    生成稳定车道线地图。
    """
    stable_lanes = []

    for lane_id in range(TARGET_LANE_COUNT):
        lane_items = []

        for record in valid_records:
            lanes = record["lanes"]
            if len(lanes) != TARGET_LANE_COUNT:
                continue
            lane_items.append(lanes[lane_id])

        if len(lane_items) < MIN_VALID_FRAMES:
            print(f"警告：lane {lane_id} 有效帧过少：{len(lane_items)}")
            continue

        coeffs = np.array([item["coeff"] for item in lane_items], dtype=np.float64)
        x_refs = np.array([item["x_ref"] for item in lane_items], dtype=np.float64)
        point_counts = np.array([item["point_count"] for item in lane_items], dtype=np.float64)

        # 稳定地图使用中位数，抗异常帧能力比平均值更好
        stable_coeff = np.median(coeffs, axis=0)

        # 用稳定投票图重新判断真实白线段分布
        paint_segments, paint_coverage, lane_type = extract_paint_segments(
            stable_coeff,
            display_candidate_mask,
            roi_mask,
            y_top,
            y_bottom,
            step=5,
        )

        x_ref = float(np.polyval(stable_coeff, y_ref))

        stable_lane = {
            "lane_id": int(lane_id),
            "coeff": [float(v) for v in stable_coeff],
            "curve_model": "x = a*y^2 + b*y + c",
            "x_ref": x_ref,
            "x_ref_std": float(np.std(x_refs)),
            "y_min": int(y_top),
            "y_max": int(y_bottom),
            "lane_type": lane_type,
            "paint_segments_y": paint_segments,
            "paint_coverage": float(paint_coverage),
            "support_frames": int(len(lane_items)),
            "median_point_count": float(np.median(point_counts)),
            "note": "boundary_curve is continuous for calculation; paint_segments_y preserves real dashed/solid display information.",
        }

        stable_lanes.append(stable_lane)

    stable_lanes = sorted(stable_lanes, key=lambda item: item["x_ref"])

    # 重新编号，保证从左到右 lane_id = 0,1,2...
    for new_id, lane in enumerate(stable_lanes):
        lane["lane_id"] = int(new_id)

    return stable_lanes


def make_stable_vote_mask(vote_count, valid_count):
    """
    多帧候选图投票。
    只保留在多个采样帧中重复出现的白线区域。
    """
    if valid_count <= 0:
        return np.zeros_like(vote_count, dtype=np.uint8)

    vote_threshold = max(2, int(valid_count * VOTE_RATIO))

    stable_mask = np.zeros_like(vote_count, dtype=np.uint8)
    stable_mask[vote_count >= vote_threshold] = 255

    return stable_mask


def draw_stable_lane_map_preview(frame, roi_points, roi_mask, stable_lanes, stable_vote_mask, y_top, y_bottom):
    """
    生成稳定车道地图预览图。
    蓝色线：计算用连续车道边界线。
    黄色线段：真实白色虚线/实线段显示层。
    红色区域：多帧稳定白线候选。
    """
    vis = frame.copy()

    # 红色显示稳定候选区域
    red_layer = np.zeros_like(frame)
    red_layer[:, :, 2] = 255
    red_blend = cv2.addWeighted(frame, 0.55, red_layer, 0.45, 0)

    mask_bool = stable_vote_mask > 0
    vis[mask_bool] = red_blend[mask_bool]

    # ROI 边框
    cv2.polylines(vis, [roi_points], True, (0, 255, 0), 2)

    # 绘制稳定车道线
    for lane in stable_lanes:
        lane_id = lane["lane_id"]
        coeff = np.array(lane["coeff"], dtype=np.float64)

        # 1. 连续计算边界线：蓝色
        continuous_pts = sample_polyline(coeff, y_top, y_bottom, roi_mask, step=3)

        if len(continuous_pts) >= 2:
            pts_np = np.array(continuous_pts, dtype=np.int32)
            cv2.polylines(
                vis,
                [pts_np],
                False,
                (255, 0, 0),
                DRAW_CONTINUOUS_THICKNESS,
            )

        # 2. 真实白线段：黄色
        for seg in lane["paint_segments_y"]:
            y1, y2 = int(seg[0]), int(seg[1])
            paint_pts = sample_polyline(coeff, y1, y2, roi_mask, step=3)

            if len(paint_pts) >= 2:
                pts_np = np.array(paint_pts, dtype=np.int32)
                cv2.polylines(
                    vis,
                    [pts_np],
                    False,
                    (0, 255, 255),
                    DRAW_PAINT_THICKNESS,
                )

        # 3. lane 编号
        label_y = int((y_top + y_bottom) * 0.62)
        label_x = int(np.polyval(coeff, label_y))

        if 0 <= label_x < frame.shape[1] and 0 <= label_y < frame.shape[0]:
            cv2.putText(
                vis,
                f"lane {lane_id}",
                (label_x + 8, label_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        vis,
        f"stable lanes: {len(stable_lanes)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


def normalize_vote_count(vote_count):
    """
    把投票图转成可视化灰度图。
    """
    if vote_count.max() <= 0:
        return np.zeros_like(vote_count, dtype=np.uint8)

    norm = vote_count.astype(np.float32) / float(vote_count.max())
    norm = (norm * 255).clip(0, 255).astype(np.uint8)

    return norm


def main():
    root_dir = cfg.project_root_from_file(__file__)

    video_path = cfg.video_path(root_dir)
    roi_config_path = cfg.roi_config_path(root_dir)

    debug_dir = cfg.debug_dir(root_dir)
    outputs_dir = cfg.outputs_dir(root_dir)

    debug_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件：{video_path}")

    if not roi_config_path.exists():
        raise FileNotFoundError(f"找不到 ROI 配置：{roi_config_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_start_frame = cfg.START_FRAME
    sample_end_frame = cfg.lane_map_sample_end_frame(fps, frame_count, sample_start_frame)
    sample_step = cfg.lane_map_sample_step_frames(fps)
    end_frame = sample_end_frame

    print("========== 稳定车道线地图构建 ==========")
    print(f"视频: {video_path}")
    print(f"分辨率: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"总帧数: {frame_count}")
    print(f"采样范围: {sample_start_frame} ~ {end_frame}")
    print(f"采样间隔: {sample_step} 帧，约 {sample_step / max(fps, 1e-6):.2f} 秒")
    print(f"目标车道边界数量: {TARGET_LANE_COUNT}")

    # 读取第一帧，用于 ROI 和预览
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_start_frame)
    ok, preview_frame = cap.read()
    if not ok:
        raise RuntimeError("无法读取预览帧")

    roi_mask, roi_points = load_roi_mask(preview_frame.shape, roi_config_path)

    y_top = int(np.min(roi_points[:, 1]))
    y_bottom = int(np.max(roi_points[:, 1]) - cfg.scale_px(25, preview_frame.shape[1], preview_frame.shape[0], min_value=8))
    y_ref = y_bottom
    display_candidate_mask = extract_lane_candidates(preview_frame, roi_mask)

    vote_count = np.zeros((height, width), dtype=np.uint16)

    valid_records = []
    all_lane_counts = []
    sampled_frames = []

    frame_indices = list(range(sample_start_frame, end_frame + 1, sample_step))

    for k, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            print(f"[{k}/{len(frame_indices)}] frame {frame_idx}: 读取失败，跳过")
            continue

        candidate_mask, lanes, info = fit_lanes_in_frame(frame, roi_mask, roi_points)

        # 累计候选图投票
        vote_count[candidate_mask > 0] += 1

        lane_count = len(lanes)
        all_lane_counts.append(lane_count)
        sampled_frames.append(frame_idx)

        status = "OK" if lane_count == TARGET_LANE_COUNT else "SKIP"

        print(
            f"[{k:03d}/{len(frame_indices):03d}] "
            f"frame={frame_idx:06d}, lanes={lane_count}, "
            f"hough={info['num_hough_models']}, status={status}"
        )

        if lane_count == TARGET_LANE_COUNT:
            valid_records.append(
                {
                    "frame_idx": int(frame_idx),
                    "lanes": lanes,
                    "num_hough_models": int(info["num_hough_models"]),
                    "num_hough_clusters": int(info["num_hough_clusters"]),
                }
            )

    cap.release()

    print("========== 采样统计 ==========")
    print(f"采样帧数量: {len(sampled_frames)}")
    print(f"有效帧数量: {len(valid_records)}")
    print(f"各帧检测 lane 数量分布: {dict(Counter(all_lane_counts))}")

    if len(valid_records) < MIN_VALID_FRAMES:
        raise RuntimeError(
            f"有效帧数量不足：{len(valid_records)}，至少需要 {MIN_VALID_FRAMES}。"
            f"可以检查 ROI 或放宽 lane_group_fit.py 中的阈值。"
        )

    stable_vote_mask = make_stable_vote_mask(vote_count, len(sampled_frames))

    stable_lanes = aggregate_lanes(
        valid_records,
        roi_mask,
        display_candidate_mask,
        y_top,
        y_bottom,
        y_ref,
    )

    if len(stable_lanes) == 0:
        raise RuntimeError("没有生成稳定车道线，请检查单帧拟合结果。")

    preview = draw_stable_lane_map_preview(
        preview_frame,
        roi_points,
        roi_mask,
        stable_lanes,
        display_candidate_mask,
        y_top,
        y_bottom,
    )

    vote_norm = normalize_vote_count(vote_count)

    stable_json_path = outputs_dir / "stable_lane_map.json"
    preview_path = debug_dir / "stable_lane_map_preview.jpg"
    vote_path = debug_dir / "stable_lane_vote_mask.jpg"

    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(vote_path), vote_norm)

    output_data = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "image_width": int(width),
        "image_height": int(height),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "sample_start_frame": int(sample_start_frame),
        "sample_end_frame": int(end_frame),
        "sample_step": int(sample_step),
        "sample_step_seconds": float(sample_step / max(fps, 1e-6)),
        "sampled_frame_count": int(len(sampled_frames)),
        "valid_frame_count": int(len(valid_records)),
        "target_lane_count": int(TARGET_LANE_COUNT),
        "detected_stable_lane_count": int(len(stable_lanes)),
        "y_top": int(y_top),
        "y_bottom": int(y_bottom),
        "y_ref": int(y_ref),
        "roi_polygon": roi_points.tolist(),
        "stable_lanes": stable_lanes,
        "note": (
            "Stable lane map built by multi-frame median fusion. "
            "Each lane has a continuous boundary curve for calculation and paint_segments_y for dashed/solid display."
        ),
    }

    with open(stable_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("========== 输出结果 ==========")
    print(f"稳定车道地图 JSON: {stable_json_path}")
    print(f"稳定车道地图预览图: {preview_path}")
    print(f"车道线投票图: {vote_path}")

    print("========== 稳定车道线 ==========")
    for lane in stable_lanes:
        print(
            f"lane {lane['lane_id']}: "
            f"x_ref={lane['x_ref']:.1f}, "
            f"type={lane['lane_type']}, "
            f"support={lane['support_frames']}, "
            f"x_ref_std={lane['x_ref_std']:.2f}, "
            f"coverage={lane['paint_coverage']:.2f}"
        )

    print("完成：稳定车道线地图已生成。")


if __name__ == "__main__":
    main()