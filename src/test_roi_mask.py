from pathlib import Path
import json
import cv2
import numpy as np
import project_config as cfg


def main():
    root_dir = cfg.project_root_from_file(__file__)

    image_path = cfg.debug_dir(root_dir) / "frame_000000.jpg"
    roi_config_path = cfg.roi_config_path(root_dir)
    output_path = cfg.debug_dir(root_dir) / "roi_masked_frame.jpg"
    mask_path = cfg.debug_dir(root_dir) / "roi_mask.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"找不到参考帧：{image_path}")

    if not roi_config_path.exists():
        raise FileNotFoundError(f"找不到 ROI 配置：{roi_config_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    with open(roi_config_path, "r", encoding="utf-8") as f:
        roi_data = json.load(f)

    points = np.array(roi_data["roi_polygon"], dtype=np.int32)

    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    masked = cv2.bitwise_and(image, image, mask=mask)

    # 为了看得更清楚，把 ROI 外区域变暗，而不是全黑
    dark = (image * 0.25).astype(np.uint8)
    preview = dark.copy()
    preview[mask > 0] = image[mask > 0]

    cv2.polylines(preview, [points], True, (0, 255, 0), 2)

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(output_path), preview)

    print("ROI 掩膜测试完成")
    print(f"ROI mask: {mask_path}")
    print(f"ROI preview: {output_path}")


if __name__ == "__main__":
    main()