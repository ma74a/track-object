# track-object

**Repo:** https://github.com/ma74a/track-object

Pick an object once with your mouse, and watch it get tracked through the rest of a video — three different implementations, from a lightweight classical tracker up to a detector-based pipeline that can recover after losing the object.

| | `only_cv2.py` | `optical_flow_cv2.py` | `with_yolo.py` |
|---|---|---|---|
| **Method** | OpenCV KCF tracker | Lucas-Kanade optical flow | YOLO + ByteTrack |
| **Recovers if lost?** | ❌ | ❌ | ✅ (IoU re-acquisition) |
| **Speed** | Fast | Fast | Slower (runs a detector every frame) |
| **Output** | Saved video file | Live preview window | Live preview window |
| **Best for** | Quick tests, static backgrounds | Understanding tracking fundamentals | Real-world video with occlusion/re-entry |

## Implementation details

**`only_cv2.py`** — the object is selected once via `cv2.selectROI`, then handed to OpenCV's `TrackerKCF`, a correlation-filter tracker that updates its position estimate frame-by-frame using appearance features. It's fast but has no way to recover once the object leaves the frame or the filter drifts off target; on failure, the frame is just tagged "Tracking lost." Output is written to disk with `cv2.VideoWriter`, matching the input's FPS and resolution.

**`optical_flow_cv2.py`** — instead of a black-box tracker, this tracks a set of good feature points (`cv2.goodFeaturesToTrack`) inside the initial ROI, then propagates them frame-to-frame with pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`). The bounding box is shifted each frame by the **median** displacement of surviving points, which makes it more robust to a few noisy/outlier points than using the mean. It's more transparent about *why* tracking fails (points literally disappear) but still has no re-acquisition logic.

**`with_yolo.py`** — the most robust of the three. A YOLO model runs detection + multi-object tracking every frame via `model.track(..., tracker="bytetrack.yaml", persist=True)`, which assigns persistent IDs across frames. The user-selected object is matched to a track ID by IoU against the initial box; if that ID disappears (e.g. after occlusion), the script re-acquires it by finding whichever current detection has the highest IoU overlap with the last known box, above a minimum threshold. This trades speed for the ability to recover after occlusion or momentary loss — something the other two approaches can't do.

## Deployment / Setup

```bash
# 1. Clone the repo
git clone https://github.com/ma74a/track-object.git
cd track-object

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`with_yolo.py` needs a YOLO weights file (e.g. `yolov8n.pt`) in `models/` — grab one from [ultralytics/assets](https://github.com/ultralytics/assets/releases), or just leave the model path pointing at a standard name (e.g. `"yolov8n.pt"`) and `ultralytics` will download it automatically on first run.

> **Note:** `only_cv2.py` uses `cv2.TrackerKCF_create()`, which lives in `opencv-contrib-python`, not plain `opencv-python`. `requirements.txt` already includes it — if you're installing packages manually, don't forget it or you'll hit an `AttributeError`.

These scripts open GUI windows (`cv2.selectROI`, `cv2.imshow`), so they need a display — they won't run headless on a server or in a container without a virtual display (e.g. `xvfb`) or code changes to skip the GUI parts.

## Usage

1. Put your video in `input_videos/`.
2. Open the script you want and set the input path (and model path, for YOLO) in the `main(...)` call at the bottom.
3. Run it:
   ```bash
   python only_cv2.py
   python optical_flow_cv2.py
   python with_yolo.py
   ```
4. On the first frame, drag a box around the object, then press **Enter** or **Space**.

`only_cv2.py` writes the tracked video to `output_videos/`. The other two open a live preview — press **q** to quit.

## Project structure

```
track-object/
├── input_videos/       # source videos
├── models/              # YOLO weights (with_yolo.py)
├── output_videos/       # tracked output (only_cv2.py)
├── only_cv2.py
├── optical_flow_cv2.py
├── with_yolo.py
└── requirements.txt
```