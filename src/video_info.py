from pathlib import Path
import json
import cv2
import project_config as cfg


def main():
    root_dir = cfg.project_root_from_file(__file__)
    video_path = cfg.video_path(root_dir)

    outputs_dir = cfg.outputs_dir(root_dir)
    debug_dir = cfg.debug_dir(root_dir)
    configs_dir = cfg.configs_dir(root_dir)

    outputs_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)
    configs_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件：{video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0

    print("========== 视频信息 ==========")
    print(f"视频路径: {video_path}")
    print(f"分辨率: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"总帧数: {frame_count}")
    print(f"时长: {duration_sec:.2f} 秒")

    meta = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }

    meta_path = outputs_dir / "video_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("无法读取视频第一帧")

    first_frame_path = debug_dir / "frame_000000.jpg"
    cv2.imwrite(str(first_frame_path), frame)

    # 保存几张参考帧，方便后面选 ROI
    sample_indices = [
        0,
        frame_count // 4,
        frame_count // 2,
        frame_count * 3 // 4,
        max(frame_count - 1, 0),
    ]

    saved_frames = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        out_path = debug_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_frames.append(str(out_path))

    cap.release()

    print("========== 输出文件 ==========")
    print(f"视频信息: {meta_path}")
    print(f"第一帧: {first_frame_path}")
    print("参考帧:")
    for p in saved_frames:
        print(f"  {p}")

    print("完成：视频读取正常。下一步可以做 ROI 区域选择。")


if __name__ == "__main__":
    main()