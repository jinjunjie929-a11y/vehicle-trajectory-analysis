from pathlib import Path
import json
import cv2
import numpy as np
import project_config as cfg


def load_roi_mask(image_shape, roi_config_path: Path):
    h, w = image_shape[:2]

    with open(roi_config_path, "r", encoding="utf-8") as f:
        roi_data = json.load(f)

    points = np.array(roi_data["roi_polygon"], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    return mask, points


def apply_roi(mask, roi_mask):
    return cv2.bitwise_and(mask, roi_mask)


def extract_lane_candidates_v2(frame, roi_mask):
    """
    高速白色车道线候选提取 V2。
    目标：先尽量提取完整车道线候选，允许少量噪声，后续再通过车道线拟合筛选。
    """

    # 轻微滤波，减少压缩噪声
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    l_channel = lab[:, :, 0]

    # 1. HSV 白色候选：低饱和 + 较高亮度
    # 这里比上一版放宽，避免远处虚线漏检
    hsv_white = cv2.inRange(
        hsv,
        np.array([0, 0, 115], dtype=np.uint8),
        np.array([180, 125, 255], dtype=np.uint8),
    )

    # 2. 灰度白色候选
    gray_white = cv2.inRange(gray, 125, 255)

    # 3. CLAHE 增强亮度通道，再做自适应阈值
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

    # 4. Top-hat 提取细长亮线
    kernel_tophat_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_tophat_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))

    tophat_1 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat_1)
    tophat_2 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat_2)

    tophat = cv2.max(tophat_1, tophat_2)
    _, tophat_bin = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)

    # 5. 综合候选
    candidate_1 = cv2.bitwise_and(hsv_white, gray_white)
    candidate_2 = cv2.bitwise_and(adaptive_white, gray_white)
    candidate_3 = cv2.bitwise_and(tophat_bin, gray_white)

    combined = cv2.bitwise_or(candidate_1, candidate_2)
    combined = cv2.bitwise_or(combined, candidate_3)

    # 限制在 ROI 内
    combined = apply_roi(combined, roi_mask)

    # 去掉图像最底部一小条，减少字幕/播放器干扰
    # 如果后续发现近处车道线不够，可以把 25 改成 0
    h_img, w_img = combined.shape[:2]
    bottom_ignore = cfg.lane_group_scaled_params(combined.shape[1], combined.shape[0])["bottom_ignore_px"]
    combined[h_img - bottom_ignore:h_img, :] = 0

    # 轻微形态学处理：不能过强，否则虚线会被破坏
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    debug_masks = {
        "hsv_white": apply_roi(candidate_1, roi_mask),
        "adaptive_white": apply_roi(candidate_2, roi_mask),
        "tophat_white": apply_roi(candidate_3, roi_mask),
        "combined_raw": combined.copy(),
    }

    return combined, debug_masks


def filter_components_v2(binary_mask):
    """
    连通域过滤 V2。
    上一版过滤太狠，这一版保留远处小虚线段。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    filtered = np.zeros_like(binary_mask)
    h_img, w_img = binary_mask.shape[:2]

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        cx, cy = centroids[label_id]

        # 远处车道线很小，所以 y 越靠上，面积阈值越低
        if cy < h_img * 0.45:
            min_area = 4
        elif cy < h_img * 0.70:
            min_area = 8
        else:
            min_area = 12

        if area < min_area:
            continue

        # 排除过大的块状区域
        if area > 12000:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side

        density = area / max(1, w * h)

        # 车道线通常是细长结构。
        # 但远处虚线可能只有几个像素，所以不能过滤太狠。
        if area >= 30:
            if aspect < 1.4 and density > 0.55:
                continue

        # 排除明显太宽的块状噪声
        if w > 220 and h > 80:
            continue

        filtered[labels == label_id] = 255

    return filtered


def make_overlay(frame, candidate_mask, roi_points):
    overlay = frame.copy()

    red_layer = np.zeros_like(frame)
    red_layer[:, :, 2] = 255

    mask_bool = candidate_mask > 0
    blended = cv2.addWeighted(frame, 0.35, red_layer, 0.65, 0)
    overlay[mask_bool] = blended[mask_bool]

    # ROI 边框仅用于调试
    cv2.polylines(overlay, [roi_points], True, (0, 255, 0), 2)

    return overlay


def to_bgr(gray_img):
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)


def make_debug_panel(frame, masks, filtered, overlay):
    h, w = frame.shape[:2]

    p1 = to_bgr(masks["hsv_white"])
    p2 = to_bgr(masks["adaptive_white"])
    p3 = to_bgr(masks["tophat_white"])
    p4 = to_bgr(filtered)

    cv2.putText(p1, "HSV + Gray", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(p2, "Adaptive", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(p3, "Tophat", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(p4, "Final Filtered", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    panel = np.vstack([top, bottom])

    scale = 0.5
    panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    overlay_small = cv2.resize(overlay, (panel.shape[1], int(panel.shape[1] * h / w)))
    final_panel = np.vstack([panel, overlay_small])

    return final_panel


def main():
    root_dir = cfg.project_root_from_file(__file__)

    frame_path = cfg.debug_dir(root_dir) / "frame_000000.jpg"
    roi_config_path = cfg.roi_config_path(root_dir)

    debug_dir = cfg.debug_dir(root_dir)
    debug_dir.mkdir(exist_ok=True)

    if not frame_path.exists():
        raise FileNotFoundError(f"找不到参考帧：{frame_path}")

    if not roi_config_path.exists():
        raise FileNotFoundError(f"找不到 ROI 配置文件：{roi_config_path}")

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"无法读取图片：{frame_path}")

    roi_mask, roi_points = load_roi_mask(frame.shape, roi_config_path)

    combined_raw, debug_masks = extract_lane_candidates_v2(frame, roi_mask)
    filtered = combined_raw.copy()
    overlay = make_overlay(frame, filtered, roi_points)
    debug_panel = make_debug_panel(frame, debug_masks, filtered, overlay)

    cv2.imwrite(str(debug_dir / "lane_v2_hsv_white.jpg"), debug_masks["hsv_white"])
    cv2.imwrite(str(debug_dir / "lane_v2_adaptive_white.jpg"), debug_masks["adaptive_white"])
    cv2.imwrite(str(debug_dir / "lane_v2_tophat_white.jpg"), debug_masks["tophat_white"])
    cv2.imwrite(str(debug_dir / "lane_v2_combined_raw.jpg"), combined_raw)
    cv2.imwrite(str(debug_dir / "lane_v2_filtered.jpg"), filtered)
    cv2.imwrite(str(debug_dir / "lane_v2_overlay.jpg"), overlay)
    cv2.imwrite(str(debug_dir / "lane_v2_debug_panel.jpg"), debug_panel)

    print("车道线候选区域提取 V2 完成")
    print(f"HSV 白线候选: {debug_dir / 'lane_v2_hsv_white.jpg'}")
    print(f"自适应白线候选: {debug_dir / 'lane_v2_adaptive_white.jpg'}")
    print(f"Tophat 白线候选: {debug_dir / 'lane_v2_tophat_white.jpg'}")
    print(f"综合候选: {debug_dir / 'lane_v2_combined_raw.jpg'}")
    print(f"过滤后候选: {debug_dir / 'lane_v2_filtered.jpg'}")
    print(f"叠加预览图: {debug_dir / 'lane_v2_overlay.jpg'}")
    print(f"调试拼图: {debug_dir / 'lane_v2_debug_panel.jpg'}")


if __name__ == "__main__":
    main()