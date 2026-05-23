"""
Chess Vision System — Intel RealSense D405
==========================================
Core module. Supports two board analysis modes:

  1. BoardAnalyzer         — uses a single global perspective transform (4-corner
                             calibration, fast, good for printed/flat boards).
  2. IrregularBoardAnalyzer — uses per-square quadrilateral regions derived from
                              9×9 = 81 grid intersection points (accurate for
                              hand-drawn or uneven boards).

run.py auto-selects the right analyzer based on what is in calibration.npz.
"""

from collections import deque
import cv2
import numpy as np
import pyrealsense2 as rs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARUCO_IDS = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Warped board resolution used by BoardAnalyzer (single-transform mode)
SQUARE_PX = 64
BOARD_PX = SQUARE_PX * 8  # 512 px

CALIBRATION_FILE = "calibration.npz"

# A piece must be this many mm CLOSER to the camera than the baseline to count
# as occupied. Since depth is now properly scaled in mm, and noise is ~±2-3mm,
# 20 mm is a very safe threshold to catch any piece taller than 2 cm.
DEFAULT_THRESHOLD_MM = 20.0

# Temporal smoothing: median-buffer this many depth frames before deciding.
# Higher = more stable but slightly more lag when placing/removing a piece.
DEFAULT_SMOOTH_FRAMES = 7

# Hysteresis: a square must read the *new* state for this many consecutive
# smoothed frames before the bitmap actually flips. Kills single-frame glitches.
DEFAULT_HYSTERESIS = 5


# ---------------------------------------------------------------------------
# Camera Stream
# ---------------------------------------------------------------------------

class CameraStream:
    """
    Wraps the RealSense D405 pipeline. Depth frames are aligned to the
    color frame so they share the same pixel coordinate space.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._align = rs.align(rs.stream.color)

    def start(self):
        profile = self.pipeline.start(self.config)
        # Query the true hardware depth scale. 
        # Standard cameras (D435) = 0.001m (1mm). D405 = 0.0001m (0.1mm).
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale_mm = depth_sensor.get_depth_scale() * 1000.0

    def stop(self):
        self.pipeline.stop()

    def get_frames(self, timeout_ms: int = 5000):
        """
        Returns (color_bgr, depth_mm) as numpy arrays, both aligned to the
        color frame. depth_mm is float32 in millimetres (0 = no data).
        Returns (None, None) if a frame is missing.
        """
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        
        color = np.asanyarray(color_frame.get_data())
        raw_depth = np.asanyarray(depth_frame.get_data())
        # Convert raw hardware units to true millimetres
        depth_mm = (raw_depth.astype(np.float32) * self.depth_scale_mm)
        
        return color, depth_mm

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Board Detector (ArUco — for 4-corner mode)
# ---------------------------------------------------------------------------

class BoardDetector:
    """
    Detects the chessboard corners using 4 ArUco markers.
    Only needed during calibration (fixed-mount usage).
    """

    def __init__(self):
        self._detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

    def detect(self, color_bgr: np.ndarray) -> dict | None:
        """
        Looks for all 4 ArUco corner markers in the image.
        Returns dict {id: (cx, cy)} if all 4 found, else None.
        """
        corners, ids, _ = self._detector.detectMarkers(color_bgr)
        if ids is None or len(ids) < 4:
            return None

        marker_centers = {}
        for i, marker_id in enumerate(ids.flatten()):
            if int(marker_id) in ARUCO_IDS:
                c = corners[i][0]
                cx, cy = c.mean(axis=0)
                marker_centers[int(marker_id)] = (float(cx), float(cy))

        if not all(k in marker_centers for k in ARUCO_IDS):
            return None
        return marker_centers

    def compute_transform(self, marker_centers: dict) -> np.ndarray:
        src = np.float32([
            marker_centers[0],
            marker_centers[1],
            marker_centers[2],
            marker_centers[3],
        ])
        dst = np.float32([
            [0,        0       ],
            [BOARD_PX, 0       ],
            [BOARD_PX, BOARD_PX],
            [0,        BOARD_PX],
        ])
        return cv2.getPerspectiveTransform(src, dst)


# ---------------------------------------------------------------------------
# Board Analyzer — single global perspective transform (4-corner calibration)
# ---------------------------------------------------------------------------

class BoardAnalyzer:
    """
    Analyzes board state using a single global perspective transform.
    Best for flat, regular boards (ArUco or 4-corner calibration).
    """

    def __init__(
        self,
        transform_M: np.ndarray,
        baseline_depth: np.ndarray,
        threshold_mm: float = DEFAULT_THRESHOLD_MM,
        smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
        hysteresis: int = DEFAULT_HYSTERESIS,
    ):
        self.M = transform_M
        self.baseline = baseline_depth
        self.threshold = threshold_mm
        self._buf = deque(maxlen=smooth_frames)
        self._state = np.zeros((8, 8), dtype=bool)      # stable confirmed state
        self._hyst  = np.zeros((8, 8), dtype=np.int8)   # consecutive-frame counter
        self._hysteresis = hysteresis
        self._detector = BoardDetector()
        self._frame_count = 0
        self.recheck_interval = 30

    def _warp(self, img: np.ndarray, flags: int = cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, (BOARD_PX, BOARD_PX), flags=flags)

    @staticmethod
    def _sample_squares(warped_depth: np.ndarray) -> np.ndarray:
        result = np.zeros((8, 8), dtype=np.float32)
        pad = int(SQUARE_PX * 0.3)
        for row in range(8):
            for col in range(8):
                y0 = row * SQUARE_PX + pad
                y1 = (row + 1) * SQUARE_PX - pad
                x0 = col * SQUARE_PX + pad
                x1 = (col + 1) * SQUARE_PX - pad
                patch = warped_depth[y0:y1, x0:x1]
                valid = patch[patch > 0]
                result[row, col] = float(np.percentile(valid, 10)) if len(valid) > 0 else 0.0
        return result

    def analyze(self, depth_mm: np.ndarray, color_bgr: np.ndarray = None, prewarped: bool = False):
        if color_bgr is not None and not prewarped:
            self._frame_count += 1
            if self._frame_count % self.recheck_interval == 0:
                corners = self._detector.detect(color_bgr)
                if corners is not None:
                    self.M = self._detector.compute_transform(corners)

        if prewarped:
            warped_depth = depth_mm
        else:
            warped_depth = self._warp(depth_mm, flags=cv2.INTER_NEAREST)
        
        # heights_full is the full resolution height map in the warped space (512x512)
        heights_full = np.zeros_like(warped_depth, dtype=np.float32)
        for r in range(8):
            for c in range(8):
                y0, y1 = r * SQUARE_PX, (r + 1) * SQUARE_PX
                x0, x1 = c * SQUARE_PX, (c + 1) * SQUARE_PX
                patch = warped_depth[y0:y1, x0:x1]
                valid = patch > 0
                if np.any(valid):
                    heights_full[y0:y1, x0:x1][valid] = self.baseline[r, c] - patch[valid]

        # Calculate percentage of pixels in each square above 2 cm (20 mm)
        percentage_grid = np.zeros((8, 8), dtype=np.float32)
        threshold_mask = np.zeros_like(warped_depth, dtype=np.uint8)
        
        pad = int(SQUARE_PX * 0.3)
        for r in range(8):
            for c in range(8):
                y0 = r * SQUARE_PX + pad
                y1 = (r + 1) * SQUARE_PX - pad
                x0 = c * SQUARE_PX + pad
                x1 = (c + 1) * SQUARE_PX - pad
                
                patch_h = heights_full[y0:y1, x0:x1]
                patch_d = warped_depth[y0:y1, x0:x1]
                valid = patch_d > 0
                if np.sum(valid) > 0:
                    above_threshold = patch_h[valid] > 20.0
                    percentage_grid[r, c] = np.sum(above_threshold) / np.sum(valid)
                    
                    # Also mark threshold mask in the whole square for better visualization
                    y0_w, y1_w = r * SQUARE_PX, (r + 1) * SQUARE_PX
                    x0_w, x1_w = c * SQUARE_PX, (c + 1) * SQUARE_PX
                    whole_h = heights_full[y0_w:y1_w, x0_w:x1_w]
                    whole_d = warped_depth[y0_w:y1_w, x0_w:x1_w]
                    threshold_mask[y0_w:y1_w, x0_w:x1_w][(whole_d > 0) & (whole_h > 20.0)] = 255
                else:
                    percentage_grid[r, c] = 0.0

        # Rolling median across last smooth_frames frames
        self._buf.append(percentage_grid)
        smooth_percentage = np.median(np.stack(self._buf, axis=0), axis=0).astype(np.float32)

        # Normalise the whole 8x8 grid such that the highest value is 1
        max_val = np.max(smooth_percentage)
        if max_val > 0.05:
            normalized_grid = smooth_percentage / max_val
        else:
            normalized_grid = np.zeros_like(smooth_percentage)

        raw_occupied = smooth_percentage > 0.50
        self._apply_hysteresis(raw_occupied)
        
        # For classification
        physical_heights = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                y0 = r * SQUARE_PX + pad
                y1 = (r + 1) * SQUARE_PX - pad
                x0 = c * SQUARE_PX + pad
                x1 = (c + 1) * SQUARE_PX - pad
                patch_h = heights_full[y0:y1, x0:x1]
                patch_d = warped_depth[y0:y1, x0:x1]
                valid = (patch_d > 0) & (patch_h > 20.0)
                if np.sum(valid) > 0:
                    physical_heights[r, c] = np.percentile(patch_h[valid], 90)
                else:
                    physical_heights[r, c] = 0.0
                    
        piece_classes = classify_pieces(physical_heights, self._state)
        return self._state.copy(), heights_full, threshold_mask, normalized_grid, piece_classes, physical_heights

    def _apply_hysteresis(self, raw: np.ndarray) -> None:
        """Only flip state after raw is consistent for self._hysteresis frames."""
        agrees = raw == self._state
        self._hyst[~agrees] += 1
        self._hyst[agrees] = 0
        flip = self._hyst >= self._hysteresis
        self._state[flip] = raw[flip]
        self._hyst[flip] = 0

    def visualize(self, color_bgr: np.ndarray, occupied_bitmap: np.ndarray) -> np.ndarray:
        warped_color = self._warp(color_bgr)
        vis = warped_color.copy()
        for row in range(8):
            for col in range(8):
                x0, y0 = col * SQUARE_PX, row * SQUARE_PX
                x1, y1 = x0 + SQUARE_PX, y0 + SQUARE_PX
                color = (0, 220, 0) if occupied_bitmap[row, col] else (60, 60, 200)
                cv2.rectangle(vis, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), color, 2)
                label = "X" if occupied_bitmap[row, col] else "."
                cv2.putText(vis, label,
                            (x0 + SQUARE_PX // 3, y0 + SQUARE_PX * 2 // 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return vis


# ---------------------------------------------------------------------------
# Irregular Board Analyzer — per-square quad regions (9×9 grid calibration)
# ---------------------------------------------------------------------------

class IrregularBoardAnalyzer:
    """
    Analyzes board state using per-square quadrilateral regions.
    Handles hand-drawn / uneven boards with non-uniform square sizes.

    grid_points : (9, 9, 2) float32 array of grid intersection (x, y) image coords.
        grid_points[row, col] is the intersection point where:
            row 0 = top boundary,  row 8 = bottom boundary
            col 0 = left boundary, col 8 = right boundary
        So square (r, c) has corners at grid_points[r,c], [r,c+1], [r+1,c+1], [r+1,c].
    """

    # Inner region factor: how much to shrink each square quad toward its centroid
    # before sampling depth. 0.3 = use inner 40% of each square.
    SHRINK = 0.3

    def __init__(
        self,
        grid_points: np.ndarray,
        baseline_depth: np.ndarray,
        threshold_mm: float = DEFAULT_THRESHOLD_MM,
        smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
        hysteresis: int = DEFAULT_HYSTERESIS,
    ):
        self.grid = grid_points.astype(np.float32)   # (9, 9, 2)
        self.baseline = baseline_depth               # (8, 8) float32
        self.threshold = threshold_mm
        self._buf = deque(maxlen=smooth_frames)
        self._state = np.zeros((8, 8), dtype=bool)      # stable confirmed state
        self._hyst  = np.zeros((8, 8), dtype=np.int8)   # consecutive-frame counter
        self._hysteresis = hysteresis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _square_quad(self, row: int, col: int) -> np.ndarray:
        """Returns (4, 2) float32 corners of square (row, col): TL, TR, BR, BL."""
        return np.array([
            self.grid[row,     col    ],
            self.grid[row,     col + 1],
            self.grid[row + 1, col + 1],
            self.grid[row + 1, col    ],
        ], dtype=np.float32)

    def _shrink_quad(self, pts: np.ndarray) -> np.ndarray:
        """Shrink quad corners toward centroid by SHRINK factor → (4,2) int32."""
        centroid = pts.mean(axis=0)
        shrunk = pts + (centroid - pts) * self.SHRINK
        return shrunk.astype(np.int32)

    def _sample_depth_in_quad(self, depth_mm: np.ndarray, quad_int32: np.ndarray) -> float:
        """10th percentile depth inside region. Better than median for small pieces."""
        h, w = depth_mm.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [quad_int32], 255)
        valid = depth_mm[mask > 0]
        valid = valid[valid > 0]
        # Use 10th percentile instead of median. If a piece only takes up 15% 
        # of the square, the median misses it completely. The 10th percentile
        # guarantees we see the height of the piece.
        return float(np.percentile(valid, 10)) if len(valid) > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API (same interface as BoardAnalyzer)
    # ------------------------------------------------------------------

    def _apply_hysteresis(self, raw: np.ndarray) -> None:
        """Only flip state after raw is consistent for self._hysteresis frames."""
        agrees = raw == self._state
        self._hyst[~agrees] += 1
        self._hyst[agrees] = 0
        flip = self._hyst >= self._hysteresis
        self._state[flip] = raw[flip]
        self._hyst[flip] = 0

    def analyze(self, depth_mm: np.ndarray, color_bgr: np.ndarray = None):
        """
        Returns (occupied_bitmap, depth_map_mm, piece_classes) arrays.
        Works directly in image coordinates — no perspective warp needed.
        Applies rolling-median smoothing and hysteresis to suppress noise.
        """
        h, w = depth_mm.shape[:2]
        
        # heights_full is the full resolution height map in image coordinates (1280x720)
        heights_full = np.zeros_like(depth_mm, dtype=np.float32)
        threshold_mask = np.zeros((h, w), dtype=np.uint8)
        percentage_grid = np.zeros((8, 8), dtype=np.float32)
        
        for r in range(8):
            for c in range(8):
                quad = self._square_quad(r, c)
                inner = self._shrink_quad(quad)
                
                # We can construct masks to isolate the pixels in each square
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [inner], 255)
                
                valid = (mask > 0) & (depth_mm > 0)
                if np.any(valid):
                    # Height = baseline - depth
                    heights_full[valid] = self.baseline[r, c] - depth_mm[valid]
                    
                    above_threshold = (heights_full > 20.0) & valid
                    threshold_mask[above_threshold] = 255
                    percentage_grid[r, c] = np.sum(above_threshold) / np.sum(valid)
                else:
                    percentage_grid[r, c] = 0.0

        # Rolling median across last smooth_frames frames
        self._buf.append(percentage_grid)
        smooth_percentage = np.median(np.stack(self._buf, axis=0), axis=0).astype(np.float32)

        # Normalise the whole 8x8 grid such that the highest value is 1
        max_val = np.max(smooth_percentage)
        if max_val > 0.05:
            normalized_grid = smooth_percentage / max_val
        else:
            normalized_grid = np.zeros_like(smooth_percentage)

        raw_occupied = smooth_percentage > 0.15
        self._apply_hysteresis(raw_occupied)
        
        # For classification
        physical_heights = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                quad = self._square_quad(r, c)
                inner = self._shrink_quad(quad)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [inner], 255)
                valid = (mask > 0) & (depth_mm > 0)
                if np.any(valid):
                    valid_heights = heights_full[valid]
                    above = valid_heights > 20.0
                    if np.any(above):
                        physical_heights[r, c] = np.percentile(valid_heights[above], 90)
                    else:
                        physical_heights[r, c] = 0.0
                        
        piece_classes = classify_pieces(physical_heights, self._state)
        return self._state.copy(), heights_full, threshold_mask, normalized_grid, piece_classes, physical_heights

    def visualize(self, color_bgr: np.ndarray, occupied_bitmap: np.ndarray) -> np.ndarray:
        """Draws per-square quads on the original (unwarped) image."""
        vis = color_bgr.copy()
        for row in range(8):
            for col in range(8):
                quad = self._square_quad(row, col).astype(np.int32)
                color = (0, 220, 0) if occupied_bitmap[row, col] else (60, 60, 200)
                cv2.polylines(vis, [quad], isClosed=True, color=color, thickness=2)
                centroid = quad.mean(axis=0).astype(int)
                label = "X" if occupied_bitmap[row, col] else "."
                cv2.putText(vis, label,
                            (centroid[0] - 8, centroid[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return vis


# ---------------------------------------------------------------------------
# Grid detection helper (auto-detect from checkerboard pattern)
# ---------------------------------------------------------------------------

def detect_grid_from_checkerboard(color_bgr: np.ndarray) -> np.ndarray | None:
    """
    Tries to auto-detect the 9×9 grid intersection points using
    cv2.findChessboardCorners on the 7×7 inner corners, then extrapolates
    to the full 9×9 boundary.

    Returns (9, 9, 2) float32 array if successful, else None.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

    # Try standard inner corners (7×7 for an 8×8 board)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (7, 7), flags)

    if not found:
        return None

    # Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    inner = corners.reshape(7, 7, 2)  # [row, col, xy]  — inner 7×7

    # Extrapolate to 9×9 boundary using linear edge extension.
    # inner[0,*] = top inner row, inner[6,*] = bottom inner row
    # Extend outward by the local edge spacing at each boundary.
    grid = np.zeros((9, 9, 2), dtype=np.float32)

    # Fill inner 7×7 into positions [1:8, 1:8]
    grid[1:8, 1:8] = inner

    # Extrapolate top row (row 0): step = inner[0] - inner[1]
    grid[0, 1:8] = inner[0] + (inner[0] - inner[1])
    # Extrapolate bottom row (row 8): step = inner[6] - inner[5]
    grid[8, 1:8] = inner[6] + (inner[6] - inner[5])

    # Extrapolate left col (col 0): step = inner[:,0] - inner[:,1]
    grid[1:8, 0] = inner[:, 0] + (inner[:, 0] - inner[:, 1])
    # Extrapolate right col (col 8): step = inner[:,6] - inner[:,5]
    grid[1:8, 8] = inner[:, 6] + (inner[:, 6] - inner[:, 5])

    # Fill the 4 outer corners by double-extrapolation
    grid[0, 0] = grid[0, 1] + (grid[1, 0] - grid[1, 1])
    grid[0, 8] = grid[0, 7] + (grid[1, 8] - grid[1, 7])
    grid[8, 0] = grid[8, 1] + (grid[7, 0] - grid[7, 1])
    grid[8, 8] = grid[8, 7] + (grid[7, 8] - grid[7, 7])

    return grid


def draw_grid(image: np.ndarray, grid: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Draws a 9×9 grid of intersection points and connecting lines on a copy of image."""
    vis = image.copy()
    pts = grid.astype(np.int32)

    # Draw horizontal lines
    for row in range(9):
        for col in range(8):
            cv2.line(vis, tuple(pts[row, col]), tuple(pts[row, col + 1]), color, 1)
    # Draw vertical lines
    for col in range(9):
        for row in range(8):
            cv2.line(vis, tuple(pts[row, col]), tuple(pts[row + 1, col]), color, 1)
    # Draw intersection dots
    for row in range(9):
        for col in range(9):
            cv2.circle(vis, tuple(pts[row, col]), 3, (0, 200, 255), -1)

    return vis


# ---------------------------------------------------------------------------
# Calibration I/O
# ---------------------------------------------------------------------------

def save_calibration_grid(
    grid_points: np.ndarray,
    baseline: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD_MM,
    path: str = CALIBRATION_FILE,
) -> None:
    """Saves 9×9 grid_points, baseline, and threshold to .npz (irregular mode)."""
    np.savez(path,
             grid_points=grid_points.astype(np.float32),
             baseline=baseline,
             threshold=np.array(threshold, dtype=np.float32))
    print(f"Calibration saved → {path}  (irregular grid mode)")


def save_calibration(
    M: np.ndarray,
    baseline: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD_MM,
    path: str = CALIBRATION_FILE,
) -> None:
    """Saves 4-corner transform matrix, baseline, and threshold to .npz."""
    np.savez(path,
             M=M,
             baseline=baseline,
             threshold=np.array(threshold, dtype=np.float32))
    print(f"Calibration saved → {path}  (4-corner transform mode)")


def load_calibration(path: str = CALIBRATION_FILE):
    """
    Loads calibration from .npz file. Auto-detects mode.

    Returns:
        If irregular grid mode : ("grid", grid_points (9,9,2), baseline, threshold)
        If 4-corner mode       : ("transform", M (3,3), baseline, threshold)
    """
    data = np.load(path)
    threshold = data["threshold"]
    if threshold.ndim == 0:
        threshold = float(threshold)
    baseline = data["baseline"]

    if "grid_points" in data:
        return "grid", data["grid_points"], baseline, threshold
    else:
        return "transform", data["M"], baseline, threshold


def classify_pieces(heights: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """
    Classifies pieces into generic categories based on their height.
    P = Pawn (< 40mm)
    M = Minor / Rook (40mm - 70mm)
    Q = Major / Queen / King (> 70mm)
    . = Empty
    """
    classes = np.full((8, 8), '.', dtype='<U1')
    classes[occupied & (heights < 40)] = 'P'
    classes[occupied & (heights >= 40) & (heights < 70)] = 'M'
    classes[occupied & (heights >= 70)] = 'Q'
    return classes
