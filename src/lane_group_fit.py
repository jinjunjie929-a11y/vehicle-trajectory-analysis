from pathlib import Path
import json
import cv2
import numpy as np
import project_config as cfg


# =========================
# 可调参数
# =========================
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH = 25
HOUGH_MAX_LINE_GAP = 90

CLUSTER_GAP_PX = 85          # Hough 线按底部 x 位置聚类的间距
POINT_DISTANCE_PX = 14       # 候选像素到车道线模型的最大距离
MIN_POINTS_PER_LANE = 80     # 每条车道线最少候选点数量
MAX_LANES = 8                # 最多保留几条车道线

DRAW_THICKNESS = 3


def load_roi_mask(image_shape, roi_config_path: Path):
    h, w = image_shape[:2]

    with open(roi_config_path, "r", encoding="utf-8") as f:
        roi_data = json.load(f)

    points = np.array(roi_data["roi_polygon"], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    return mask, points


def extract_lane_candidates(frame, roi_mask):
    """
    与前面 V2 基本一致：
    先尽量把白色车道线候选提全，后面再通过拟合筛选。
    """
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)

    l_channel = lab[:, :, 0]

    hsv_white = cv2.inRange(
        hsv,
        np.array([0, 0, 115], dtype=np.uint8),
        np.array([180, 125, 255], dtype=np.uint8),
    )

    gray_white = cv2.inRange(gray, 125, 255)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    adaptive_white = cv2.adaptiveThreshold(
        l_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        -8,
    )

    kernel_tophat_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_tophat_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))

    tophat_1 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat_1)
    tophat_2 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat_2)
    tophat = cv2.max(tophat_1, tophat_2)

    _, tophat_bin = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)

    candidate_1 = cv2.bitwise_and(hsv_white, gray_white)
    candidate_2 = cv2.bitwise_and(adaptive_white, gray_white)
    candidate_3 = cv2.bitwise_and(tophat_bin, gray_white)

    combined = cv2.bitwise_or(candidate_1, candidate_2)
    combined = cv2.bitwise_or(combined, candidate_3)
    combined = cv2.bitwise_and(combined, roi_mask)

    # 去掉最底部，减少文字/播放器干扰
    h_img, _ = combined.shape[:2]
    bottom_ignore = cfg.lane_group_scaled_params(combined.shape[1], combined.shape[0])["bottom_ignore_px"]
    combined[h_img - bottom_ignore:h_img, :] = 0

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return combined


def detect_hough_line_models(candidate_mask):
    """
    提取 Hough 线段，并转成 x = m*y + b 的形式。
    """
    edges = cv2.Canny(candidate_mask, 50, 150)

    h_img, w_img = candidate_mask.shape[:2]
    params = cfg.lane_group_scaled_params(w_img, h_img)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=params["hough_threshold"],
        minLineLength=params["hough_min_line_length"],
        maxLineGap=params["hough_max_line_gap"],
    )

    if lines is None:
        return []

    models = []

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))

        if length < params["hough_min_line_length"]:
            continue

        # 避免近似水平线
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < 8:
            continue

        # 避免 dy 太小导致 x = m*y + b 不稳定
        if abs(dy) < 8:
            continue

        m = dx / dy
        b = x1 - m * y1

        models.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "m": float(m),
                "b": float(b),
                "length": float(length),
                "angle": float(angle),
            }
        )

    return models


def cluster_hough_models(models, y_ref):
    """
    按每条 Hough 线延长到 y_ref 时的 x 位置进行聚类。
    同一条车道线的多个线段应聚到一起。
    """
    if not models:
        return []

    for model in models:
        model["x_ref"] = model["m"] * y_ref + model["b"]

    models_sorted = sorted(models, key=lambda item: item["x_ref"])

    clusters = []

    current = [models_sorted[0]]
    current_mean = models_sorted[0]["x_ref"]

    for item in models_sorted[1:]:
        width_est = max(1, int(max(abs(m.get("x_ref", 0)) for m in models_sorted) * 1.25))
        cluster_gap_px = cfg.scale_px(CLUSTER_GAP_PX, width_est, 720, min_value=30)
        if abs(item["x_ref"] - current_mean) <= cluster_gap_px:
            current.append(item)
            current_mean = float(np.mean([m["x_ref"] for m in current]))
        else:
            clusters.append(current)
            current = [item]
            current_mean = item["x_ref"]

    clusters.append(current)

    return clusters


def merge_cluster_to_line_model(cluster):
    """
    将一个 Hough 聚类合成一个粗略线模型。
    用长度加权，长线段权重大。
    """
    lengths = np.array([item["length"] for item in cluster], dtype=np.float32)
    weights = lengths / max(float(lengths.sum()), 1.0)

    m = float(np.sum([item["m"] * w for item, w in zip(cluster, weights)]))
    b = float(np.sum([item["b"] * w for item, w in zip(cluster, weights)]))
    x_ref = float(np.sum([item["x_ref"] * w for item, w in zip(cluster, weights)]))

    return {
        "m": m,
        "b": b,
        "x_ref": x_ref,
        "num_segments": len(cluster),
        "total_length": float(lengths.sum()),
    }


def collect_points_near_line(candidate_mask, line_model, roi_mask):
    """
    收集距离粗略线模型较近的候选像素点。
    """
    ys, xs = np.where((candidate_mask > 0) & (roi_mask > 0))

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    m = line_model["m"]
    b = line_model["b"]

    x_pred = m * ys + b
    dist = np.abs(xs - x_pred)

    h_img, w_img = candidate_mask.shape[:2]
    point_distance_px = cfg.lane_group_scaled_params(w_img, h_img)["point_distance_px"]
    keep = dist <= point_distance_px

    pts = np.column_stack([xs[keep], ys[keep]]).astype(np.float32)

    return pts


def fit_quadratic_x_by_y(points):
    """
    拟合 x = a*y^2 + b*y + c。
    """
    if len(points) == 0:
        return None
    w_est = max(int(np.max(points[:, 0]) + 1), cfg.BASE_WIDTH)
    h_est = max(int(np.max(points[:, 1]) + 1), cfg.BASE_HEIGHT)
    params = cfg.lane_group_scaled_params(w_est, h_est)
    min_points_per_lane = params["min_points_per_lane"]
    point_distance_px = params["point_distance_px"]
    residual_min_px = params["residual_min_px"]

    if len(points) < min_points_per_lane:
        return None

    xs = points[:, 0]
    ys = points[:, 1]

    # 二次拟合
    coeff = np.polyfit(ys, xs, deg=2)

    # 粗略剔除离群点后再拟合一次
    x_fit = np.polyval(coeff, ys)
    residual = np.abs(xs - x_fit)

    good = residual <= max(point_distance_px * 1.5, residual_min_px)

    if good.sum() < min_points_per_lane:
        return None

    coeff = np.polyfit(ys[good], xs[good], deg=2)

    return coeff, points[good]


def sample_polyline(coeff, y_min, y_max, roi_mask, step=4):
    """
    根据拟合曲线采样点，只保留 ROI 内的点。
    """
    h, w = roi_mask.shape[:2]

    y_values = np.arange(int(y_min), int(y_max) + 1, step)
    pts = []

    for y in y_values:
        x = np.polyval(coeff, y)
        x_i = int(round(x))
        y_i = int(round(y))

        if 0 <= x_i < w and 0 <= y_i < h and roi_mask[y_i, x_i] > 0:
            pts.append([x_i, y_i])

    return pts


def extract_paint_segments(coeff, candidate_mask, roi_mask, y_min, y_max, step=5):
    """
    沿拟合曲线检查哪些位置真实存在白色车道线。
    这个用于区分“显示层虚线段”和“计算层连续线”。
    """
    h, w = candidate_mask.shape[:2]

    visible_samples = []

    for y in range(int(y_min), int(y_max) + 1, step):
        x = np.polyval(coeff, y)
        x_i = int(round(x))
        y_i = int(round(y))

        if not (0 <= x_i < w and 0 <= y_i < h):
            visible_samples.append(False)
            continue

        if roi_mask[y_i, x_i] == 0:
            visible_samples.append(False)
            continue

        params = cfg.lane_group_scaled_params(w, h)
        px = params["paint_patch_x"]
        py = params["paint_patch_y"]
        x1 = max(0, x_i - px)
        x2 = min(w, x_i + px + 1)
        y1 = max(0, y_i - py)
        y2 = min(h, y_i + py + 1)

        patch = candidate_mask[y1:y2, x1:x2]
        has_paint = int(np.count_nonzero(patch)) >= params["paint_min_pixels"]

        visible_samples.append(has_paint)

    y_values = list(range(int(y_min), int(y_max) + 1, step))

    segments = []
    in_seg = False
    seg_start_y = None

    for y, flag in zip(y_values, visible_samples):
        if flag and not in_seg:
            in_seg = True
            seg_start_y = y

        elif not flag and in_seg:
            in_seg = False
            seg_end_y = y - step

            if seg_end_y - seg_start_y >= cfg.scale_px(8, w, h, min_value=4):
                segments.append([int(seg_start_y), int(seg_end_y)])

    if in_seg:
        seg_end_y = y_values[-1]
        if seg_end_y - seg_start_y >= cfg.scale_px(8, w, h, min_value=4):
            segments.append([int(seg_start_y), int(seg_end_y)])

    coverage = float(np.mean(visible_samples)) if visible_samples else 0.0

    lane_type = "solid" if coverage > 0.55 and len(segments) <= 2 else "dashed"

    return segments, coverage, lane_type


def merge_duplicate_lanes(lanes, y_ref):
    """
    去除重复车道线。按 y_ref 处 x 位置排序，太近的合并。
    """
    if not lanes:
        return []

    for lane in lanes:
        lane["x_ref"] = float(np.polyval(lane["coeff"], y_ref))

    lanes = sorted(lanes, key=lambda item: item["x_ref"])

    merged = []

    for lane in lanes:
        if not merged:
            merged.append(lane)
            continue

        prev = merged[-1]

        width_est = max(1, int(max(abs(lane["x_ref"]), abs(prev["x_ref"])) * 1.25))
        duplicate_gap_px = cfg.scale_px(45, width_est, 720, min_value=18)
        if abs(lane["x_ref"] - prev["x_ref"]) < duplicate_gap_px:
            # 两条太近，保留点数更多的一条
            if lane["point_count"] > prev["point_count"]:
                merged[-1] = lane
        else:
            merged.append(lane)

    return merged


def draw_lanes(frame, candidate_mask, roi_points, lanes, roi_mask, y_top, y_bottom):
    vis = frame.copy()

    # 红色显示候选区域
    red_layer = np.zeros_like(frame)
    red_layer[:, :, 2] = 255

    blended = cv2.addWeighted(frame, 0.45, red_layer, 0.55, 0)
    mask_bool = candidate_mask > 0
    vis[mask_bool] = blended[mask_bool]

    # ROI 边框
    cv2.polylines(vis, [roi_points], True, (0, 255, 0), 2)

    # 绘制拟合后的连续车道边界线
    for lane in lanes:
        coeff = lane["coeff"]
        lane_id = lane["lane_id"]

        pts = sample_polyline(coeff, y_top, y_bottom, roi_mask, step=3)

        if len(pts) >= 2:
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(vis, [pts_np], False, (255, 0, 0), cfg.scale_px(DRAW_THICKNESS, frame.shape[1], frame.shape[0], min_value=1))

            # 半透明青色再叠一层，方便看
            overlay = vis.copy()
            cv2.polylines(overlay, [pts_np], False, (255, 255, 0), 1)
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

            # 标注 lane id
            mid_idx = len(pts) // 2
            tx, ty = pts[mid_idx]
            cv2.putText(
                vis,
                f"lane {lane_id}",
                (tx + 8, ty - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        vis,
        f"detected lanes: {len(lanes)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


def main():
    root_dir = Path(__file__).resolve().parents[1]

    frame_path = root_dir / "debug" / "frame_000000.jpg"
    roi_config_path = root_dir / "configs" / "roi_config.json"

    debug_dir = root_dir / "debug"
    outputs_dir = root_dir / "outputs"

    debug_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(f"无法读取图片：{frame_path}")

    roi_mask, roi_points = load_roi_mask(frame.shape, roi_config_path)

    y_top = int(np.min(roi_points[:, 1]))
    y_bottom = int(np.max(roi_points[:, 1]) - 25)
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

        # 如果采样范围太短，说明不是稳定车道线
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
                "paint_coverage": coverage,
                "lane_type": lane_type,
            }
        )

    lanes = merge_duplicate_lanes(lanes, y_ref)
    lanes = lanes[:MAX_LANES]

    # 按 y_ref 位置从左到右重新编号
    for i, lane in enumerate(lanes):
        lane["lane_id"] = i
        lane["coeff"] = [float(v) for v in lane["coeff"]]
        lane["x_ref"] = float(np.polyval(lane["coeff"], y_ref))

    result_img = draw_lanes(
        frame,
        candidate_mask,
        roi_points,
        lanes,
        roi_mask,
        y_top,
        y_bottom,
    )

    candidate_path = debug_dir / "lane_group_candidate_mask.jpg"
    result_path = debug_dir / "lane_group_fit_result.jpg"
    json_path = outputs_dir / "lane_group_fit_single_frame.json"

    cv2.imwrite(str(candidate_path), candidate_mask)
    cv2.imwrite(str(result_path), result_img)

    output_data = {
        "source_frame": str(frame_path),
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "y_top": int(y_top),
        "y_bottom": int(y_bottom),
        "y_ref": int(y_ref),
        "num_hough_models": int(len(hough_models)),
        "num_hough_clusters": int(len(clusters)),
        "num_lanes": int(len(lanes)),
        "lanes": lanes,
        "note": "Single-frame lane grouping and quadratic fitting. Boundary curve is for calculation; paint_segments_y preserves dashed/solid display information.",
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("========== 单帧车道线分组拟合完成 ==========")
    print(f"Hough 线段数量: {len(hough_models)}")
    print(f"Hough 聚类数量: {len(clusters)}")
    print(f"拟合车道线数量: {len(lanes)}")
    print(f"候选二值图: {candidate_path}")
    print(f"拟合结果图: {result_path}")
    print(f"拟合结果 JSON: {json_path}")

    for lane in lanes:
        print(
            f"lane {lane['lane_id']}: "
            f"x_ref={lane['x_ref']:.1f}, "
            f"type={lane['lane_type']}, "
            f"points={lane['point_count']}, "
            f"coverage={lane['paint_coverage']:.2f}"
        )


if __name__ == "__main__":
    main()