"""
Detect Peaks Service — RealSense Chess Vision
====================================
Service module that captures a specified number of frames,
detects peaks, reduces them to 1 per square per frame,
and outputs filtered occupancy and XYZ position maps.

Can be easily called by other software (e.g. ROS2 node).
"""

import argparse
import cv2
import json
import numpy as np
import pyrealsense2 as rs
from chess_vision import BOARD_PX, CameraStream, load_calibration

def warp_board(image: np.ndarray, matrix: np.ndarray, size: int, flags: int) -> np.ndarray:
    return cv2.warpPerspective(image, matrix, (size, size), flags=flags)

def grid_corners(grid: np.ndarray) -> np.ndarray:
    return np.float32([grid[0, 0], grid[0, 8], grid[8, 8], grid[8, 0]])

def build_baseline_map(baseline: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(baseline, (size, size), interpolation=cv2.INTER_NEAREST)

def compute_height_map(warped_depth: np.ndarray, baseline_map: np.ndarray) -> np.ndarray:
    height = np.zeros_like(warped_depth, dtype=np.float32)
    valid = warped_depth > 0
    height[valid] = baseline_map[valid] - warped_depth[valid]
    height[height < 0] = 0.0
    return height

def sample_depth_mm(depth_mm: np.ndarray, u: float, v: float, window: int = 5) -> float:
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or vi < 0 or ui >= depth_mm.shape[1] or vi >= depth_mm.shape[0]:
        return 0.0
    r = max(1, window // 2)
    x0 = max(0, ui - r)
    x1 = min(depth_mm.shape[1], ui + r + 1)
    y0 = max(0, vi - r)
    y1 = min(depth_mm.shape[0], vi + r + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if len(valid) else 0.0

def classify_piece(height_mm: float) -> str:
    classes = {
        "Pawn": 45.0,
        "Rook": 50.0,
        "Bish/Kni": 65.0,
        "King/Que": 90.0
    }
    return min(classes.keys(), key=lambda k: abs(classes[k] - height_mm))

def find_peaks(
    height_map: np.ndarray,
    min_height: float,
    max_peaks: int,
) -> list[tuple[int, int, float]]:
    """Finds peaks in the height map without limiting per-square."""
    if height_map.size == 0:
        return []
    blurred = cv2.GaussianBlur(height_map, (0, 0), 1.2)
    kernel = np.ones((9, 9), dtype=np.uint8)
    local_max = cv2.dilate(blurred, kernel)
    peak_mask = (blurred >= (local_max - 1e-3)) & (blurred >= min_height)
    peak_mask = peak_mask.astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(peak_mask, connectivity=8)
    peaks = []
    for label in range(1, num):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        vals = blurred[ys, xs]
        idx = int(np.argmax(vals))
        x, y = int(xs[idx]), int(ys[idx])
        peaks.append((x, y, float(height_map[y, x])))

    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks[:max_peaks]

def remove_outliers(points: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Removes outliers using the Modified Z-Score method based on Median Absolute Deviation."""
    if len(points) <= 2:
        return points
    d = np.abs(points - np.median(points, axis=0))
    mdev = np.median(d, axis=0)
    mdev[mdev == 0] = 1e-6  # Prevent division by zero
    s = d / mdev
    valid = np.all(s < m, axis=1)
    if np.any(valid):
        return points[valid]
    return points

def detect_peaks_service(num_frames: int = 100, calibration_path: str = "calibration.npz", min_height: float = 35.0, clamp_board: bool = True):
    """
    Runs peak detection for a specified number of frames and returns averaged results.
    
    Args:
        num_frames: Number of frames to process.
        calibration_path: Path to the camera calibration file.
        min_height: Minimum height threshold for piece detection.
        clamp_board: If True, clamps peaks slightly outside the board into the nearest square.
        
    Returns:
        occupancy_map: np.ndarray of shape (8, 8) with values in [0.0, 1.0] representing
                       the percentage of frames a piece was detected in each square.
        peak_positions: np.ndarray of shape (8, 8, 3) representing the averaged XYZ
                        position (in meters, camera frame) of the peak in each square. 
                        Squares without pieces will have [0, 0, 0].
        piece_classes: np.ndarray of shape (8, 8) containing the predicted piece class
                       strings ("Pawn", "Rook", etc) based on the peak height. Empty string
                       if no piece was detected.
    """
    mode, data, baseline, _ = load_calibration(calibration_path)

    if mode == "grid":
        grid_points = data.astype(np.float32)
        src = grid_corners(grid_points)
        dst = np.float32(
            [
                [0, 0],
                [BOARD_PX, 0],
                [BOARD_PX, BOARD_PX],
                [0, BOARD_PX],
            ]
        )
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = data.astype(np.float32)

    M_inv = np.linalg.inv(M)
    baseline_map = build_baseline_map(baseline.astype(np.float32), BOARD_PX)

    BUFFER_PX = 40
    WARPED_SIZE = BOARD_PX + 2 * BUFFER_PX
    T = np.array([[1, 0, BUFFER_PX], [0, 1, BUFFER_PX], [0, 0, 1]], dtype=np.float32)
    M_buf = T @ M
    M_buf_inv = np.linalg.inv(M_buf)
    baseline_map_padded = cv2.copyMakeBorder(baseline_map, BUFFER_PX, BUFFER_PX, BUFFER_PX, BUFFER_PX, cv2.BORDER_REPLICATE)

    # Dictionary to collect XYZ points for each square over the captured frames
    history = {(r, c): [] for r in range(8) for c in range(8)}

    cam = CameraStream()
    with cam:
        profile = cam.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()

        frames_processed = 0
        while frames_processed < num_frames:
            color, depth = cam.get_frames()
            if color is None or depth is None:
                continue

            warped_depth = warp_board(depth, M_buf, WARPED_SIZE, flags=cv2.INTER_NEAREST)
            height_map = compute_height_map(warped_depth, baseline_map_padded)
            
            # Find all peaks
            peaks = find_peaks(height_map, min_height=min_height, max_peaks=128)

            if peaks:
                pts = np.float32([[p[0], p[1]] for p in peaks]).reshape(-1, 1, 2)
                pts_raw = cv2.perspectiveTransform(pts, M_buf_inv).reshape(-1, 2)
            else:
                pts_raw = np.empty((0, 2), dtype=np.float32)

            world_xy_mm = []
            world_xyz = []
            valid_peak_indices = []
            
            for i, (u, v) in enumerate(pts_raw):
                ui, vi = int(round(u)), int(round(v))
                if ui < 0 or vi < 0 or ui >= depth.shape[1] or vi >= depth.shape[0]:
                    continue
                d_mm = sample_depth_mm(depth, u, v)
                if d_mm <= 0:
                    continue
                xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], d_mm / 1000.0)
                world_xy_mm.append((xyz[0] * 1000.0, -xyz[1] * 1000.0))
                # Store full xyz in meters (or whatever format user expects).
                world_xyz.append(np.array([xyz[0], xyz[1], xyz[2]]))
                valid_peak_indices.append(i)

            # Get world corners
            corners_warped = np.float32(
                [
                    [0.0, 0.0],
                    [BOARD_PX - 1.0, 0.0],
                    [BOARD_PX - 1.0, BOARD_PX - 1.0],
                    [0.0, BOARD_PX - 1.0],
                ]
            ).reshape(-1, 1, 2)
            corners_raw = cv2.perspectiveTransform(corners_warped, M_inv).reshape(4, 2)
            corner_world = []
            for u, v in corners_raw:
                ui, vi = int(round(u)), int(round(v))
                if ui < 0 or vi < 0 or ui >= depth.shape[1] or vi >= depth.shape[0]:
                    corner_world = []
                    break
                d_mm = sample_depth_mm(depth, u, v)
                if d_mm <= 0:
                    corner_world = []
                    break
                xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], d_mm / 1000.0)
                corner_world.append((xyz[0] * 1000.0, -xyz[1] * 1000.0))

            frame_square_xyz = {}

            if len(corner_world) == 4:
                tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in corner_world]
                uvec = (tr - tl)[:2]
                vvec = (bl - tl)[:2]
                A = np.array([[uvec[0], vvec[0]], [uvec[1], vvec[1]]], dtype=np.float32)
                det = float(np.linalg.det(A))
                if abs(det) > 1e-6:
                    A_inv = np.linalg.inv(A)
                    
                    # Group XYZ points by square using 3D world coordinates
                    square_xyz_list = {}
                    
                    for i, (xw, yw) in enumerate(world_xy_mm):
                        rel = np.array([xw - tl[0], yw - tl[1]], dtype=np.float32)
                        s, t = A_inv @ rel
                        
                        if clamp_board:
                            # Clamp within reasonable bounds (-20% to 120%) avoiding wild noise
                            if -0.2 < s < 1.2 and -0.2 < t < 1.2:
                                s = max(0.0, min(0.999, float(s)))
                                t = max(0.0, min(0.999, float(t)))

                        if 0.0 <= s < 1.0 and 0.0 <= t < 1.0:
                            c = int(s * 8)
                            r = int(t * 8)
                            if (r, c) not in square_xyz_list:
                                square_xyz_list[(r, c)] = []
                            square_xyz_list[(r, c)].append(world_xyz[i])
                    
                    for (r, c), xyz_list in square_xyz_list.items():
                        frame_square_xyz[(r, c)] = np.mean(xyz_list, axis=0)
            
            if not frame_square_xyz and world_xyz:
                # Fallback to UV-based assignment if corners can't be deprojected
                cell = BOARD_PX / 8.0
                square_xyz_list = {}
                for i, idx in enumerate(valid_peak_indices):
                    x, y, _ = peaks[idx]
                    c = int((x - BUFFER_PX) / cell) if cell > 0 else 0
                    r = int((y - BUFFER_PX) / cell) if cell > 0 else 0
                    r = max(0, min(7, r))
                    c = max(0, min(7, c))
                    if (r, c) not in square_xyz_list:
                        square_xyz_list[(r, c)] = []
                    square_xyz_list[(r, c)].append(world_xyz[i])
                
                for (r, c), xyz_list in square_xyz_list.items():
                    frame_square_xyz[(r, c)] = np.mean(xyz_list, axis=0)

            # 2. Accumulate history across frames
            for (r, c), xyz in frame_square_xyz.items():
                history[(r, c)].append(xyz)
                
            frames_processed += 1

    # 3. Filter output across frames and compute final maps
    occupancy_map = np.zeros((8, 8), dtype=np.float32)
    peak_positions = np.zeros((8, 8, 3), dtype=np.float32)
    piece_classes = np.empty((8, 8), dtype=object)
    piece_classes.fill("")

    for r in range(8):
        for c in range(8):
            pts = history[(r, c)]
            occ_ratio = len(pts) / num_frames
            occupancy_map[r, c] = occ_ratio
            
            if len(pts) > 0:
                pts_arr = np.array(pts)
                
                # Remove outliers based on spatial distribution
                filtered_pts = remove_outliers(pts_arr)
                
                # Average out the remaining valid points per square
                avg_xyz = np.mean(filtered_pts, axis=0)
                peak_positions[r, c] = avg_xyz
                
                # Use the Z coordinate (height) to classify the piece
                piece_classes[r, c] = classify_piece(abs(avg_xyz[2]) * 1000.0)

    return occupancy_map, peak_positions, piece_classes


def visualize_output(occupancy_map: np.ndarray, peak_positions: np.ndarray, piece_classes: np.ndarray):
    """Renders a visualization of the 8x8 occupancy map and Z-heights."""
    cell_size = 70
    img = np.zeros((cell_size * 8, cell_size * 8, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            occ = occupancy_map[r, c]
            
            # Checkerboard background (dark gray / light gray)
            bg_color = (60, 60, 60) if (r + c) % 2 == 0 else (100, 100, 100)
            
            # Draw cell background
            cv2.rectangle(img, (c * cell_size, r * cell_size), ((c + 1) * cell_size, (r + 1) * cell_size), bg_color, -1)
            
            if occ > 0:
                # Color based on occupancy: 0 = dark green, 1 = bright green
                color = (0, int(50 + occ * 205), 0)
                
                # Draw occupancy circle
                radius = int(occ * (cell_size / 2 - 8))
                if radius > 0:
                    cv2.circle(img, (int(c * cell_size + cell_size / 2), int(r * cell_size + cell_size / 2)), radius, color, -1)
                
                # Draw Z value text (in mm for readability)
                z_mm = peak_positions[r, c, 2] * 1000.0
                text = f"{z_mm:.0f}mm"
                pt_class = piece_classes[r, c]
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Draw class text
                class_size = cv2.getTextSize(pt_class, font, 0.4, 1)[0]
                class_x = int(c * cell_size + (cell_size - class_size[0]) / 2)
                class_y = int(r * cell_size + cell_size / 2 - 4)
                cv2.putText(img, pt_class, (class_x, class_y), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, pt_class, (class_x, class_y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw height text
                text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                text_x = int(c * cell_size + (cell_size - text_size[0]) / 2)
                text_y = int(r * cell_size + cell_size / 2 + 12)
                
                # Add a black outline to text for contrast
                cv2.putText(img, text, (text_x, text_y), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (text_x, text_y), font, 0.4, (200, 255, 200), 1, cv2.LINE_AA)

    cv2.imshow("Final Output (Occupancy & Z height)", img)
    print("Press any key in the visualization window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detect peaks service.")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to process")
    # True by default, --no-visualize to turn off
    parser.add_argument("--visualize", action="store_true", default=True, help="Visualize the final output (default: True)")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize", help="Disable visualization")
    parser.add_argument("--no-clamp", action="store_false", dest="clamp_board", help="Disable clamping of XYZ peaks to [0, 1] board coordinates")
    parser.add_argument("--calibration", type=str, default="calibration.npz", help="Path to calibration file")
    parser.add_argument("--json-out", type=str, default="", help="Path to write JSON output")
    args = parser.parse_args()

    print(f"Running detect_peaks_service for {args.num_frames} frames...")
    occupancy, positions, classes = detect_peaks_service(num_frames=args.num_frames, calibration_path=args.calibration, clamp_board=args.clamp_board)
    
    print("\nOccupancy Map (8x8) [0.0 to 1.0]:")
    print(np.round(occupancy, 2))
    
    print("\nSample Peak Position at [0,0] (XYZ in meters):")
    print(positions[0, 0])
    
    print("\nSample Piece Class at [0,0]:")
    print(classes[0, 0])

    if args.json_out:
        out_data = {
            "occupancy": occupancy.tolist(),
            "positions": positions.tolist(),
            "classes": classes.tolist()
        }
        with open(args.json_out, "w") as f:
            json.dump(out_data, f)
        print(f"\nSaved JSON output to {args.json_out}")

    if args.visualize:
        visualize_output(occupancy, positions, classes)
