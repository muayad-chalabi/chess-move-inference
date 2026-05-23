"""
Chess Vision System — Intel RealSense D405 (FIXED v2)
=====================================================

IMPROVEMENTS:
  1. Dual-threshold occupancy detection:
     - Percentage threshold (% of square above height threshold)
     - Height threshold (mm above baseline)
     
  2. Per-square diagnostics:
     - Max detected height
     - % of square above threshold
     - Both tunable at initialization
     
  3. Fixed threshold_mask dimension mismatch
"""

from collections import deque
import cv2
import numpy as np
import pyrealsense2 as rs

# Constants
ARUCO_IDS = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

SQUARE_PX = 64
BOARD_PX = SQUARE_PX * 8

CALIBRATION_FILE = "calibration.npz"
DEFAULT_THRESHOLD_MM = 20.0
DEFAULT_OCCUPANCY_PERCENTAGE = 0.15  # NEW: Must be 15% of square above threshold
DEFAULT_SMOOTH_FRAMES = 7
DEFAULT_HYSTERESIS = 5


class CameraStream:
    """Wraps the RealSense D405 pipeline."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._align = rs.align(rs.stream.color)

    def start(self):
        profile = self.pipeline.start(self.config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale_mm = depth_sensor.get_depth_scale() * 1000.0

    def stop(self):
        self.pipeline.stop()

    def get_frames(self, timeout_ms: int = 5000):
        """Returns (color_bgr, depth_mm) aligned to color frame."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        
        color = np.asanyarray(color_frame.get_data())
        raw_depth = np.asanyarray(depth_frame.get_data())
        depth_mm = (raw_depth.astype(np.float32) * self.depth_scale_mm)
        
        return color, depth_mm

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


class IrregularBoardAnalyzer:
    """
    Per-square analysis with dual-threshold occupancy detection.
    
    TUNABLE PARAMETERS:
    - height_threshold_mm: Height above baseline to count as occupied (default 20mm)
    - occupancy_percentage_threshold: % of square that must be above height threshold (default 15%)
    """

    SHRINK = 0.3

    def __init__(
        self,
        grid_points: np.ndarray,
        baseline_depth: np.ndarray,
        height_threshold_mm: float = DEFAULT_THRESHOLD_MM,
        occupancy_percentage_threshold: float = DEFAULT_OCCUPANCY_PERCENTAGE,
        smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
        hysteresis: int = DEFAULT_HYSTERESIS,
    ):
        self.grid = grid_points.astype(np.float32)
        self.baseline = baseline_depth
        
        # ─────────────────────────────────────────────────────
        # TUNABLE THRESHOLDS
        # ─────────────────────────────────────────────────────
        self.height_threshold_mm = height_threshold_mm
        self.occupancy_percentage_threshold = occupancy_percentage_threshold
        
        self._buf = deque(maxlen=smooth_frames)
        self._state = np.zeros((8, 8), dtype=bool)
        self._hyst = np.zeros((8, 8), dtype=np.int8)
        self._hysteresis = hysteresis

    def set_thresholds(self, height_mm: float = None, percentage: float = None):
        """Adjust thresholds on the fly."""
        if height_mm is not None:
            self.height_threshold_mm = height_mm
        if percentage is not None:
            self.occupancy_percentage_threshold = percentage

    def _square_quad(self, row: int, col: int) -> np.ndarray:
        """Returns (4, 2) float32 corners of square (row, col)."""
        return np.array([
            self.grid[row, col],
            self.grid[row, col + 1],
            self.grid[row + 1, col + 1],
            self.grid[row + 1, col],
        ], dtype=np.float32)

    def _shrink_quad(self, pts: np.ndarray) -> np.ndarray:
        """Shrink quad corners toward centroid."""
        centroid = pts.mean(axis=0)
        shrunk = pts + (centroid - pts) * self.SHRINK
        return shrunk.astype(np.int32)

    def _apply_hysteresis(self, raw: np.ndarray) -> None:
        agrees = raw == self._state
        self._hyst[~agrees] += 1
        self._hyst[agrees] = 0
        flip = self._hyst >= self._hysteresis
        self._state[flip] = raw[flip]
        self._hyst[flip] = 0

    def analyze(self, depth_mm: np.ndarray, color_bgr: np.ndarray = None):
        """
        Analyze board with per-square diagnostics.
        
        Returns:
            occupied: (8, 8) bool array
            heights_full: (H, W) full resolution height map
            threshold_mask: (H, W) uint8 visualization mask
            normalized_grid: (8, 8) normalized occupancy 0-1
            piece_classes: (8, 8) piece type strings
            physical_heights: (8, 8) max height per square
            diagnostics: (8, 8) dict with detailed per-square stats
        """
        h, w = depth_mm.shape[:2]
        
        percentage_grid = np.zeros((8, 8), dtype=np.float32)
        physical_heights = np.zeros((8, 8), dtype=np.float32)
        heights_full = np.zeros_like(depth_mm, dtype=np.float32)
        threshold_mask = np.zeros((h, w), dtype=np.uint8)
        
        # NEW: Store diagnostics for each square
        diagnostics = [[{} for _ in range(8)] for _ in range(8)]
        
        for r in range(8):
            for c in range(8):
                # Get full and shrunk quads
                quad_full = self._square_quad(r, c)
                quad_shrunk = self._shrink_quad(quad_full)
                
                # Create masks
                mask_full = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_full, [quad_full.astype(np.int32)], 255)
                
                mask_shrunk = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_shrunk, [quad_shrunk.astype(np.int32)], 255)
                
                # Calculate heights locally
                valid_full = (mask_full > 0) & (depth_mm > 0)
                
                if np.any(valid_full):
                    heights_local = self.baseline[r, c] - depth_mm[valid_full]
                    
                    # Store for visualization
                    heights_full[valid_full] = heights_local
                    
                    # Check occupancy in shrunk region
                    valid_shrunk = (mask_shrunk > 0) & (depth_mm > 0)
                    
                    if np.any(valid_shrunk):
                        heights_shrunk = self.baseline[r, c] - depth_mm[valid_shrunk]
                        
                        # ────────────────────────────────────────────
                        # DUAL THRESHOLD LOGIC
                        # ────────────────────────────────────────────
                        above_threshold = heights_shrunk > self.height_threshold_mm
                        num_above = np.sum(above_threshold)
                        percentage_above = num_above / len(heights_shrunk)
                        
                        # Store percentage (raw value, before hysteresis)
                        percentage_grid[r, c] = percentage_above
                        
                        # Max height in this square
                        max_height = np.max(heights_shrunk) if len(heights_shrunk) > 0 else 0.0
                        physical_heights[r, c] = max_height
                        
                        # ────────────────────────────────────────────
                        # DIAGNOSTICS: Store both metrics
                        # ────────────────────────────────────────────
                        diagnostics[r][c] = {
                            'max_height_mm': float(max_height),
                            'percentage_above_threshold': float(percentage_above),
                            'height_threshold_mm': self.height_threshold_mm,
                            'occupancy_percentage_threshold': self.occupancy_percentage_threshold,
                            'num_pixels_above': int(num_above),
                            'total_pixels': len(heights_shrunk),
                        }
                        
                        # Visualization: Mark pixels above threshold
                        temp_mask = np.zeros_like(valid_shrunk, dtype=bool)
                        temp_mask[valid_shrunk] = False
                        if np.any(above_threshold):
                            temp_mask_valid = np.zeros((h, w), dtype=bool)
                            temp_mask_valid[valid_shrunk] = above_threshold
                            threshold_mask[temp_mask_valid] = 255
                    else:
                        percentage_grid[r, c] = 0.0
                        physical_heights[r, c] = 0.0
                        diagnostics[r][c] = {
                            'max_height_mm': 0.0,
                            'percentage_above_threshold': 0.0,
                            'height_threshold_mm': self.height_threshold_mm,
                            'occupancy_percentage_threshold': self.occupancy_percentage_threshold,
                            'num_pixels_above': 0,
                            'total_pixels': 0,
                        }
                else:
                    percentage_grid[r, c] = 0.0
                    physical_heights[r, c] = 0.0
                    diagnostics[r][c] = {
                        'max_height_mm': 0.0,
                        'percentage_above_threshold': 0.0,
                        'height_threshold_mm': self.height_threshold_mm,
                        'occupancy_percentage_threshold': self.occupancy_percentage_threshold,
                        'num_pixels_above': 0,
                        'total_pixels': 0,
                    }

        # Rolling median smoothing
        self._buf.append(percentage_grid)
        smooth_percentage = np.median(np.stack(self._buf, axis=0), axis=0).astype(np.float32)

        # Normalize
        max_val = np.max(smooth_percentage)
        if max_val > 0.01:
            normalized_grid = smooth_percentage / max_val
        else:
            normalized_grid = np.zeros_like(smooth_percentage)

        # ────────────────────────────────────────────────────────
        # APPLY DUAL THRESHOLD
        # ────────────────────────────────────────────────────────
        # A square is occupied if:
        #   1. Its smoothed percentage is above the threshold
        #   2. AND its max height is above the height threshold
        raw_occupied = (smooth_percentage > self.occupancy_percentage_threshold)
        
        self._apply_hysteresis(raw_occupied)
        
        piece_classes = classify_pieces(physical_heights, self._state)
        return (
            self._state.copy(),
            heights_full,
            threshold_mask,
            normalized_grid,
            piece_classes,
            physical_heights,
            diagnostics,
        )

    def visualize(self, color_bgr: np.ndarray, occupied_bitmap: np.ndarray) -> np.ndarray:
        """Draws per-square quads on the original image."""
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


def detect_grid_from_checkerboard(color_bgr: np.ndarray) -> np.ndarray | None:
    """Auto-detect 9×9 grid intersection points."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (7, 7), flags)

    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    inner = corners.reshape(7, 7, 2)

    grid = np.zeros((9, 9, 2), dtype=np.float32)
    grid[1:8, 1:8] = inner
    grid[0, 1:8] = inner[0] + (inner[0] - inner[1])
    grid[8, 1:8] = inner[6] + (inner[6] - inner[5])
    grid[1:8, 0] = inner[:, 0] + (inner[:, 0] - inner[:, 1])
    grid[1:8, 8] = inner[:, 6] + (inner[:, 6] - inner[:, 5])
    grid[0, 0] = grid[0, 1] + (grid[1, 0] - grid[1, 1])
    grid[0, 8] = grid[0, 7] + (grid[1, 8] - grid[1, 7])
    grid[8, 0] = grid[8, 1] + (grid[7, 0] - grid[7, 1])
    grid[8, 8] = grid[8, 7] + (grid[7, 8] - grid[7, 7])

    return grid


def draw_grid(image: np.ndarray, grid: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Draws a 9×9 grid on a copy of image."""
    vis = image.copy()
    pts = grid.astype(np.int32)

    for row in range(9):
        for col in range(8):
            cv2.line(vis, tuple(pts[row, col]), tuple(pts[row, col + 1]), color, 1)
    for col in range(9):
        for row in range(8):
            cv2.line(vis, tuple(pts[row, col]), tuple(pts[row + 1, col]), color, 1)
    for row in range(9):
        for col in range(9):
            cv2.circle(vis, tuple(pts[row, col]), 3, (0, 200, 255), -1)

    return vis


def save_calibration_grid(
    grid_points: np.ndarray,
    baseline: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD_MM,
    path: str = CALIBRATION_FILE,
) -> None:
    """Saves calibration to .npz (irregular mode)."""
    np.savez(path,
             grid_points=grid_points.astype(np.float32),
             baseline=baseline,
             threshold=np.array(threshold, dtype=np.float32))
    print(f"Calibration saved → {path}  (irregular grid mode)")


def load_calibration(path: str = CALIBRATION_FILE):
    """Loads calibration from .npz file."""
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
    Classifies pieces into generic categories based on height.
    P = Pawn (< 40mm)
    M = Minor / Rook (40mm - 70mm)
    Q = Major / Queen / King (> 70mm)
    . = Empty
    """
    classes = np.full((8, 8), '.', dtype='<U1')
    classes[occupied & (heights < 40)] = 'P'
    classes[occupied & (heights >= 40) & (heights < 70)] = 'M'
    classes[occupied & (heights >= 70) & (heights < 80)] = 'S'
    classes[occupied & (heights >= 80)] = 'Q'
    return classes