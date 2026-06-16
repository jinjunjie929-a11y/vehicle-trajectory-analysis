from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import project_config as cfg  # noqa: E402


def rel(path: Path) -> str:
    """Return a readable project-relative path for logs."""
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    """Create runtime folders used by the pipeline."""
    cfg.ensure_project_dirs(ROOT_DIR)


def missing_files(paths: Iterable[Path]) -> list[Path]:
    """Collect inputs that must exist before a step can run."""
    return [path for path in paths if not path.exists()]


def run_script(script_name: str, args: list[str] | None = None) -> None:
    """Run one script in src/ and stop the pipeline if it fails."""
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path), *(args or [])]
    print(f"\n$ {' '.join(command)}")
    start = time.time()
    result = subprocess.run(command, cwd=str(ROOT_DIR))
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

    print(f"[OK] {script_name} finished in {elapsed:.1f}s")


def run_checked_step(
    title: str,
    script_name: str,
    required_inputs: Iterable[Path],
    expected_outputs: Iterable[Path],
    script_args: list[str] | None = None,
    check_only: bool = False,
) -> None:
    """Validate inputs, execute one pipeline step, and print expected outputs."""
    print("\n" + "=" * 80)
    print(title)

    missing = missing_files(required_inputs)
    if missing:
        print("Missing required input:")
        for path in missing:
            print(f"  - {rel(path)}")

        if cfg.roi_config_path(ROOT_DIR) in missing:
            print("\nCreate ROI config first:")
            print("  python main.py --prepare-frame")
            print("  python src/roi_picker.py")

        if check_only:
            print("[CHECK] Step skipped because inputs are not available yet.")
            return

        raise FileNotFoundError(f"Cannot run step: {title}")

    if check_only:
        print("[CHECK] Inputs are ready; step not executed.")
    else:
        run_script(script_name, script_args)

    print("Expected output:")
    for path in expected_outputs:
        status = "exists" if path.exists() else "missing"
        print(f"  [{status}] {rel(path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle detection, ROI filtering, trajectory export, and trajectory plotting pipeline."
    )
    parser.add_argument("--prepare-frame", action="store_true", help="Only export video metadata and the first frame for ROI selection.")
    parser.add_argument("--pick-roi", action="store_true", help="Open the interactive ROI picker after preparing the first frame.")
    parser.add_argument("--skip-roi-test", action="store_true", help="Skip ROI mask validation.")
    parser.add_argument("--skip-lane-map", action="store_true", help="Skip stable lane map generation.")
    parser.add_argument("--skip-detection", action="store_true", help="Skip vehicle detection and distance measurement.")
    parser.add_argument("--skip-trajectory-excel", action="store_true", help="Skip ROI-relative Excel trajectory export.")
    parser.add_argument("--skip-trajectory-plots", action="store_true", help="Skip trajectory image/video rendering.")
    parser.add_argument("--no-trajectory-video", action="store_true", help="Render trajectory images only, without the trajectory overlay video.")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and show outputs without running scripts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    video_path = cfg.video_path(ROOT_DIR)
    roi_path = cfg.roi_config_path(ROOT_DIR)
    lane_map_path = cfg.lane_map_path(ROOT_DIR)
    model_path = cfg.model_path(ROOT_DIR)
    distance_csv = cfg.outputs_dir(ROOT_DIR) / cfg.RIGHT_DISTANCE_CSV_NAME
    distance_video = cfg.outputs_dir(ROOT_DIR) / cfg.RIGHT_DISTANCE_VIDEO_NAME
    trajectory_xlsx = cfg.outputs_dir(ROOT_DIR) / cfg.TRAJECTORY_XLSX_NAME
    trajectory_overview = cfg.outputs_dir(ROOT_DIR) / cfg.TRAJECTORY_OVERVIEW_NAME
    trajectory_by_id_dir = cfg.outputs_dir(ROOT_DIR) / cfg.TRAJECTORY_BY_ID_DIR_NAME
    trajectory_render_dir = cfg.outputs_dir(ROOT_DIR) / cfg.TRAJECTORY_RENDER_DIR_NAME

    print("=" * 80)
    print("Vehicle trajectory analysis pipeline")
    print("=" * 80)
    print(f"Project root: {ROOT_DIR}")
    print(f"Config: {rel(ROOT_DIR / 'config.yaml')}")
    print(f"Input video: {rel(video_path)}")
    print(f"YOLO model: {rel(model_path)}")
    print(f"ROI config: {rel(roi_path)}")

    # Step 1: read video metadata and save a reference frame for ROI selection.
    run_checked_step(
        title="1. Prepare video metadata and ROI reference frame",
        script_name="video_info.py",
        required_inputs=[video_path],
        expected_outputs=[cfg.outputs_dir(ROOT_DIR) / "video_meta.json", cfg.debug_dir(ROOT_DIR) / "frame_000000.jpg"],
        check_only=args.check_only,
    )

    if args.prepare_frame and not args.pick_roi:
        return

    if args.pick_roi:
        # ROI selection is interactive, so it is controlled by an explicit flag.
        run_script("roi_picker.py")
        if args.prepare_frame:
            return

    # Step 2: validate that the selected ROI covers the expected road area.
    if not args.skip_roi_test:
        run_checked_step(
            title="2. Validate ROI mask",
            script_name="test_roi_mask.py",
            required_inputs=[cfg.debug_dir(ROOT_DIR) / "frame_000000.jpg", roi_path],
            expected_outputs=[cfg.debug_dir(ROOT_DIR) / "roi_mask.jpg", cfg.debug_dir(ROOT_DIR) / "roi_masked_frame.jpg"],
            check_only=args.check_only,
        )

    # Step 3: build a stable lane map from sampled frames inside the ROI.
    if not args.skip_lane_map:
        run_checked_step(
            title="3. Build stable lane map",
            script_name="build_stable_lane_map.py",
            required_inputs=[video_path, roi_path],
            expected_outputs=[lane_map_path, cfg.debug_dir(ROOT_DIR) / "stable_lane_map_preview.jpg"],
            check_only=args.check_only,
        )

    # Step 4: detect vehicles with YOLO and record per-frame distance/trajectory data.
    if not args.skip_detection:
        run_checked_step(
            title="4. Detect vehicles and record trajectory CSV",
            script_name="measure_vehicle_right_distance.py",
            required_inputs=[video_path, lane_map_path, model_path],
            expected_outputs=[distance_csv, distance_video],
            check_only=args.check_only,
        )

    # Step 5: convert image-coordinate tracks into ROI-relative XY coordinates and export Excel.
    if not args.skip_trajectory_excel:
        run_checked_step(
            title="5. Export ROI-relative trajectory Excel",
            script_name="export_vehicle_trajectory_xy.py",
            required_inputs=[distance_csv, roi_path],
            expected_outputs=[trajectory_xlsx, trajectory_overview, trajectory_by_id_dir],
            script_args=[
                "--csv",
                str(distance_csv),
                "--roi",
                str(roi_path),
                "--out-xlsx",
                str(trajectory_xlsx),
                "--overview",
                str(trajectory_overview),
                "--by-id-dir",
                str(trajectory_by_id_dir),
            ],
            check_only=args.check_only,
        )

    # Step 6: render trajectory overview, per-vehicle plots, summary CSV, and optional overlay video.
    if not args.skip_trajectory_plots:
        render_args = [
            "--csv",
            str(distance_csv),
            "--video",
            str(video_path),
            "--output-dir",
            str(trajectory_render_dir),
        ]
        if args.no_trajectory_video:
            render_args.append("--no-video")

        run_checked_step(
            title="6. Render trajectory plots",
            script_name="render_vehicle_trajectories.py",
            required_inputs=[distance_csv],
            expected_outputs=[
                trajectory_render_dir / "vehicle_trajectory_overview.png",
                trajectory_render_dir / "vehicle_trajectory_summary.csv",
                trajectory_render_dir / "vehicle_trajectory_by_id",
            ],
            script_args=render_args,
            check_only=args.check_only,
        )

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
