# RealSense Chess Vision System

This project implements a computer vision pipeline using an **Intel RealSense D405** depth camera to monitor a chessboard. The system detects piece occupancy and estimates piece types by comparing real-time depth data against an empty-board baseline.

The camera is intended to be mounted looking down at a hand-drawn or standard chessboard. The pipeline atomically outputs 8x8 NumPy arrays representing the board's depth map and occupancy status, which are consumed by an external chess engine.

## Key Features

- **Sub-millimeter Depth Accuracy**: Designed specifically around the RealSense D405's high-precision 0.1mm depth scale.
- **Irregular Board Support**: Perfectly maps hand-drawn or uneven boards via a user-defined 9x9 click grid.
- **Piece Height Segmentation**: Automatically classifies pieces into categories (`P` Pawn, `M` Minor/Rook, `Q` Major/King) based on depth delta.
- **Adaptive Thresholding**: Dynamically computes a noise threshold per-square ($\theta = \mu_{noise} + k \cdot \sigma_{noise}$) during calibration to handle varying depth accuracy across the lens.
- **Temporal Hysteresis & Smoothing**: Utilizes median-frame buffers and state hysteresis to completely eliminate single-frame glitches while the robot arm hovers.
- **ArUco Re-registration**: For flat/printed boards, the system periodically checks ArUco markers to dynamically adjust the perspective warp if the board is bumped.

## Architecture

1. **Calibration (`calibrate.py`)**: 
   Since the board is generally fixed relative to the camera, calibration is run once while an empty board region is selected. 
   - Uses `matplotlib` to capture user clicks to define the board boundaries.
   - Captures the empty board baseline and calculates the per-square adaptive noise threshold.
   - Outputs `calibration.npz`.

2. **Core Vision (`chess_vision.py`)**:
   - Manages the RealSense pipeline, aligning the depth stream to the color stream.
   - Includes `IrregularBoardAnalyzer` for grid-based boards and `BoardAnalyzer` for ArUco 4-corner mode.

3. **Execution Loop (`run.py`)**:
   - Runs continuously, polling the camera at 30 FPS to keep the smoothing buffers saturated.
   - Prints a live console UI with piece categories and atomically saves arrays for external processes.

## Quick Start Guide

### Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install pyrealsense2 opencv-python numpy matplotlib
```

### Step 1: Calibration
Before detecting pieces, the system needs a depth baseline for the board. Make sure at least one clear, empty region of the board is visible (no pieces), and your RealSense camera is plugged in.
```bash
python3 calibrate.py
```
- A window will appear showing the live feed.
- The system will try to auto-detect corners (cyan dots). 
- Left-click on the grid intersections to snap to them. You don't need to click all 81; click the outer corners and a scattering of inner ones, and the system will extrapolate the rest.
- Close the window to confirm. Then drag to select an empty board region; the script will capture 40 frames to build the `baseline` and adaptive `threshold` map, saving them to `calibration.npz`.

### Step 2: Live Detection
Once calibrated, you can start the detection loop. Place some pieces on the board and run:
```bash
python3 run.py
```
- The terminal will display a live view of the board, updating every second.
- You will see pieces classified as `P`, `M`, or `Q` based on their height.
- Output arrays (`occupied_bitmap.npy` and `depth_map_mm.npy`) will be written to the current directory atomically.

**Options:**
- `--once`: Capture a single frame, print, save, and exit.
- `--interval 0.5`: Change the save/print frequency to every 0.5 seconds.
- `--visualize`: Opens a live OpenCV window showing the camera feed with the square bounding boxes overlaid.
## Technical Challenges & Solutions

During development, we encountered and solved several complex issues:

### 1. OpenCV Wayland/Qt Backend Crashing
* **Problem**: Running `calibrate.py` on Linux with GNOME/Wayland caused a `NULL window handler in cvSetMouseCallback` error. The Qt backend failed to map the window to the display server before we attached mouse events.
* **Solution**: Bypassed OpenCV's `highgui` window entirely for the calibration steps. We rewrote the calibration UI using `matplotlib` with the `TkAgg` backend, providing a stable, cross-platform click interface.

### 2. Uneven Hand-Drawn Board
* **Problem**: The original implementation used a 4-corner perspective warp (`cv2.getPerspectiveTransform`). Because the user's board was hand-drawn, the internal squares were not perfectly uniform, causing the warped grid to miss the physical squares.
* **Solution**: Created `IrregularBoardAnalyzer`. Instead of a global warp, the user clicks (or the system auto-detects) all 81 grid intersections. Each of the 64 squares is defined as a unique polygon, perfectly matching the drawn board.

### 3. Flickering and Sensor Noise
* **Problem**: Depth sensor noise (±2-3mm) caused the occupancy state of squares to rapidly flicker between occupied and empty on a frame-by-frame basis.
* **Solution**: 
  - Introduced a **rolling median buffer** (last 7 frames) to smooth out high-frequency noise.
  - Added a **hysteresis filter** (requires 3 consecutive smoothed frames to agree before flipping the state).

### 4. Asynchronous Loop Lag
* **Problem**: The detection loop in `run.py` had a `time.sleep(1.0)` to limit output to 1 update per second. Because the camera was only polled once per second, the 10-frame buffer took 10 seconds to fill, causing massive lag in piece detection.
* **Solution**: Separated the camera polling from the output saving. The `while True` loop now pulls frames at 30 FPS to keep the buffer saturated, but uses a timer to only `atomic_save` and print once per second.

### 5. Intel RealSense D405 Depth Scale Unit
* **Problem**: Standard RealSense cameras (D435/D415) report depth in 1.0 mm units. The D405 is a sub-millimeter camera that reports depth in **0.1 mm units**. Our depth matrix thought the board was 2.6 meters away (2600 units) instead of 26 cm.
* **Solution**: Updated `CameraStream` to query the physical hardware scale (`depth_sensor.get_depth_scale()`) dynamically and multiply the raw frame by `scale * 1000.0`, resulting in a true millimeter float32 array.

### 6. Small Pieces vs. Square Area (The "Median" Bug)
* **Problem**: If a small piece (like a plastic cone) only occupied 15% of a square's pixels, taking the `np.median()` depth of that square would return the depth of the empty board beneath it, causing the piece to be completely ignored.
* **Solution**: Switched the spatial sampling from `np.median` to the **10th percentile** (`np.percentile(..., 10)`). This ensures the system looks at the 10% *closest* pixels in the square, reliably capturing the height of the piece regardless of its footprint. (Note: Currently investigating final tweaks as testing continues).
