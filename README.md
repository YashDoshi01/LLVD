# LLVD — Lattice Layer Vehicle Detection

A classical computer-vision traffic analysis system that detects vehicles, estimates density, tracks objects, counts crossings, and estimates speed — all without deep learning.

## Project Structure

```
LLVD/
├── run.py                  # 🚀 Entry point — start here
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── config/                 # Configuration files
│   ├── default.json        #   Default config (MVI_39761 image sequence)
│   └── sample_video.json   #   Example for .mp4 video files
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── advanced_pipeline.py  # Full pipeline: tracking + counting + speed + fog/night adapt
│   └── base_pipeline.py      # Base pipeline: DBSCAN clustering + bounding boxes
│
├── data/                   # Place your video files or image sequence folders here
│
├── output/                 # Generated output
│   ├── videos/             #   Annotated output videos
│   └── logs/               #   Profiling logs (JSON)
│
└── tests/                  # Unit tests (to be added)
```

## Quick Start

### 1. Prerequisites

- **Python 3.8+**
- A traffic video file (`.mp4`, `.avi`) or an image sequence folder (folder of `.jpg` files)

### 2. Setup

```bash
# Navigate to the project
cd LLVD

# Create virtual environment (recommended)
python -m venv venv
venv/Scripts/activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

Edit `config/default.json` to point to your data:

```json
{
  "video": "../MVI_39761/MVI_39761",   ← path to image folder or .mp4 file
  "color_channel": "gray",             ← "gray", "V", "H", "S", "H+S+V", etc.
  "grids": { "rows": 7, "cols": 14 },  ← grid resolution for the lattice layer
  "rois": {
    "roi1": { "x": 545, "y": 159, "w": 284, "h": 140 },  ← lane 1 region
    "roi2": { "x": 238, "y": 161, "w": 284, "h": 140 }   ← lane 2 region
  }
}
```

> **Important**: The `video` path is relative to the `LLVD/` root directory.
> For image sequences, point to the folder containing the `.jpg` files.

### 4. Run

```bash
# Run the advanced pipeline (tracking + counting + speed)
python run.py

# Run with a specific config
python run.py --config config/default.json

# Run the base pipeline (DBSCAN clustering, simpler output)
python run.py --pipeline base

# See all options
python run.py --help
```

### 5. Output

After processing completes, you'll find:

| Output | Location |
|--------|----------|
| Annotated video | `output/videos/` |
| Profile logs | `output/logs/` |
| Console stats | Printed to terminal (density, counts, speed, scene conditions) |

---

## Pipelines

### Advanced Pipeline (`--pipeline advanced`, default)

Full-featured traffic analysis:
- **Scene adaptation** — auto-detects day/dusk/night/fog conditions
- **CLAHE histogram equalization** — adaptive contrast for varying light
- **Grid-based motion detection** — high-performance vectorized processing
- **Centroid tracker** — persistent object IDs across frames
- **Counting line** — counts vehicles crossing a virtual line
- **Speed estimation** — pixel-displacement-based speed in km/h
- **Async video writer** — non-blocking output
- **Multiprocessing** — uses all CPU cores

### Base Pipeline (`--pipeline base`)

Simplified version:
- Grid-based motion detection
- DBSCAN clustering of active cells
- Bounding box generation
- Density calculation
- Multiprocessing support

---

## Configuration Reference

| Field | Type | Description |
|-------|------|-------------|
| `video` | string | Path to video file or image sequence folder |
| `color_channel` | string | `"gray"`, `"H"`, `"S"`, `"V"`, `"H+S"`, `"H+V"`, `"S+V"`, `"H+S+V"` |
| `grids.rows` | int | Number of rows in the detection grid |
| `grids.cols` | int | Number of columns in the detection grid |
| `rois.roi1` | object | `{x, y, w, h}` — pixel coordinates of lane 1 ROI |
| `rois.roi2` | object | `{x, y, w, h}` — pixel coordinates of lane 2 ROI |
| `batch_size` | int | Frames per processing batch (default: 64) |

---

## How to Find ROI Coordinates for Your Video

1. Open your video in any image viewer or run:
   ```bash
   python -c "import cv2; f=cv2.imread('data/your_frame.jpg'); print(f.shape)"
   ```
2. Note the frame dimensions (height × width).
3. Identify the lanes you want to monitor.
4. Use an image editor (Paint, GIMP) to find the `(x, y)` of the top-left corner and the `(w, h)` of each lane region.
5. Update `config/your_config.json` with those values.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn` |
| `Cannot open video` | Check the `video` path in config — must be relative to `LLVD/` |
| `Frames are black / no detections` | ROI coordinates may not match your video. See "How to Find ROI Coordinates" above. |
| `RuntimeError: freeze_support()` | On Windows, make sure you're running via `python run.py`, not importing directly |
