from pathlib import Path
import json
import cv2
import numpy as np
import project_config as cfg


class ROIPicker:
    def __init__(self, image_path: Path, save_path: Path, preview_path: Path):
        self.image_path = image_path
        self.save_path = save_path
        self.preview_path = preview_path

        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(f"无法读取图片：{image_path}")

        self.display = self.image.copy()
        self.points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.redraw()

    def redraw(self):
        self.display = self.image.copy()

        # 画已经选择的点
        for i, p in enumerate(self.points):
            cv2.circle(self.display, tuple(p), 5, (0, 0, 255), -1)
            cv2.putText(
                self.display,
                str(i + 1),
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # 连线
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(self.display, [pts], False, (0, 255, 255), 2)

        # 闭合多边形预览
        if len(self.points) >= 3:
            overlay = self.display.copy()
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            self.display = cv2.addWeighted(overlay, 0.25, self.display, 0.75, 0)

            cv2.polylines(self.display, [pts], True, (0, 255, 0), 2)

        help_text = [
            "Left click: add point",
            "Right click: undo point",
            "S: save ROI",
            "R: reset",
            "ESC/Q: quit",
        ]

        y0 = 30
        for text in help_text:
            cv2.putText(
                self.display,
                text,
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y0 += 28

    def run(self):
        self.redraw()

        window_name = "ROI Picker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        h, w = self.image.shape[:2]
        cv2.resizeWindow(window_name, min(1280, w), min(720, h))
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("========== ROI 选择说明 ==========")
        print("鼠标左键：添加 ROI 点")
        print("鼠标右键：撤销上一个点")
        print("按 S：保存 ROI")
        print("按 R：重置")
        print("按 ESC 或 Q：退出")
        print("建议只框选右侧高速主车道区域，排除天空、树林、隔离带、水印、底部遮挡。")

        while True:
            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key in [27, ord("q"), ord("Q")]:
                print("退出，未保存。")
                break

            elif key in [ord("r"), ord("R")]:
                self.points = []
                self.redraw()
                print("ROI 已重置。")

            elif key in [ord("s"), ord("S")]:
                if len(self.points) < 3:
                    print("至少需要选择 3 个点。")
                    continue

                self.save_roi()
                print(f"ROI 已保存：{self.save_path}")
                print(f"ROI 预览图已保存：{self.preview_path}")
                break

        cv2.destroyAllWindows()

    def save_roi(self):
        h, w = self.image.shape[:2]

        roi_data = {
            "image_width": w,
            "image_height": h,
            "roi_polygon": self.points,
            "note": "ROI polygon for target highway lane area",
        }

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(roi_data, f, ensure_ascii=False, indent=2)

        preview = self.image.copy()
        pts = np.array(self.points, dtype=np.int32)

        overlay = preview.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
        cv2.polylines(preview, [pts], True, (0, 255, 0), 2)

        for i, p in enumerate(self.points):
            cv2.circle(preview, tuple(p), 5, (0, 0, 255), -1)
            cv2.putText(
                preview,
                str(i + 1),
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(self.preview_path), preview)


def main():
    root_dir = cfg.project_root_from_file(__file__)

    image_path = cfg.debug_dir(root_dir) / "frame_000000.jpg"
    save_path = cfg.roi_config_path(root_dir)
    preview_path = cfg.debug_dir(root_dir) / "roi_preview.jpg"

    picker = ROIPicker(image_path, save_path, preview_path)
    picker.run()


if __name__ == "__main__":
    main()