# Source Modules

This directory contains the executable modules used by `main.py`.

| Module | Responsibility |
| --- | --- |
| `project_config.py` | Loads `config.yaml` and exposes backward-compatible settings for the pipeline scripts. |
| `video_info.py` | Reads video metadata and exports the first frame for ROI selection. |
| `roi_picker.py` | Interactive ROI selection tool. |
| `test_roi_mask.py` | Validates ROI mask coverage on the reference frame. |
| `lane_candidate_extractor.py` | Extracts lane-line candidate pixels from ROI frames. |
| `lane_group_fit.py` | Groups and fits lane-line candidates. |
| `build_stable_lane_map.py` | Aggregates sampled frames into a stable lane map. |
| `measure_vehicle_right_distance.py` | Runs YOLO detection and writes per-frame vehicle trajectory/distance records. |
| `export_vehicle_trajectory_xy.py` | Converts tracked points to ROI-relative coordinates and exports Excel/plots. |
| `render_vehicle_trajectories.py` | Renders trajectory overview, per-vehicle plots, and optional overlay video. |

`main.py` is the recommended entry point. Running modules directly is useful for debugging a single step.
