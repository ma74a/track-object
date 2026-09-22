# track-object

Single-object tracking in video, implemented three different ways — from a classical tracker, to optical flow, to a detector + tracker pipeline. You pick the object once with your mouse, and the script follows it through the rest of the video.

## Methods

| Script | Approach | Notes |
|---|---|---|
| `only_cv2.py` | OpenCV's built-in KCF tracker | Fast and simple, but drifts or fails on occlusion/fast motion. Writes result to a video file. |
| `optical_flow_cv2.py` | Lucas-Kanade optical flow on feature points inside the ROI | Tracks by median displacement of good features. Shows a live preview window. |
| `with_yolo.py` | YOLO detection + ByteTrack, with IoU-based re-acquisition | Most robust — if the tracked ID is lost, it re-finds the object by matching IoU against the last known box. |

## Requirements

```bash
pip install opencv-python ultralytics numpy
```

`with_yolo.py` additionally needs a YOLO weights file (e.g. `yolov8n.pt`) placed in `models/`.

## Usage

1. Put your input video in `input_videos/`.
2. Edit the `main(...)` call at the bottom of the script you want to run (input path, and model path for the YOLO version).
3. Run it:

```bash
python only_cv2.py
python optical_flow_cv2.py
python with_yolo.py
```

4. A window will pop up on the first frame — drag a box around the object you want to track, then press Enter/Space.

- `only_cv2.py` writes the tracked video to `output_videos/`.
- `optical_flow_cv2.py` and `with_yolo.py` show a live preview; press `q` to quit early.

## Project structure

```
track-object/
├── input_videos/     # source videos
├── models/            # YOLO weights (for with_yolo.py)
├── output_videos/     # tracked output (only_cv2.py)
├── runs/               # ultralytics run artifacts
├── tests/
├── only_cv2.py
├── optical_flow_cv2.py
└── with_yolo.py
```

## Status

Exploratory comparison of tracking approaches — no formal evaluation (e.g. IoU-vs-ground-truth, FPS benchmarking) yet.