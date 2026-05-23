"""
Detect Peaks — RealSense Chess Vision
====================================
Streams RGB + depth, applies a perspective transform to flatten the board,
builds a height map (baseline - depth), detects height peaks, and converts
peak UV coordinates to XYZ (camera-centered) world coordinates.

Usage:
    python3 detect_peaks.py
    python3 detect_peaks.py --calibration calibration.npz
    python3 detect_peaks.py --min-height 25 --max-fps 10
"""

import argparse
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from chess_vision import BOARD_PX, CameraStream, load_calibration


def warp_board(
    image: np.ndarray, matrix: np.ndarray, size: int, flags: int
) -> np.ndarray:
    return cv2.warpPerspective(image, matrix, (size, size), flags=flags)


def grid_corners(grid: np.ndarray) -> np.ndarray:
    return np.float32([grid[0, 0], grid[0, 8], grid[8, 8], grid[8, 0]])


def build_baseline_map(baseline: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(baseline, (size, size), interpolation=cv2.INTER_NEAREST)


def compute_height_map(
    warped_depth: np.ndarray, baseline_map: np.ndarray
) -> np.ndarray:
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


def find_peaks(
    height_map: np.ndarray,
    min_height: float,
    max_peaks: int,
    max_per_square: int = 2,
) -> list[tuple[int, int, float]]:
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

    if max_per_square > 0:
        h, w = height_map.shape[:2]
        cell_w = w / 8.0
        cell_h = h / 8.0
        per_square: dict[tuple[int, int], int] = {}
        limited: list[tuple[int, int, float]] = []
        for x, y, height in peaks:
            c = int(x / cell_w) if cell_w > 0 else 0
            r = int(y / cell_h) if cell_h > 0 else 0
            r = max(0, min(7, r))
            c = max(0, min(7, c))
            key = (r, c)
            count = per_square.get(key, 0)
            if count < max_per_square:
                per_square[key] = count + 1
                limited.append((x, y, height))
        peaks = limited

    return peaks[:max_peaks]


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    vis = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    valid = depth_mm > 0
    if np.any(valid):
        d_min = float(np.min(depth_mm[valid]))
        d_max = float(np.max(depth_mm[valid]))
        if d_max > d_min:
            norm = (depth_mm - d_min) / (d_max - d_min)
            norm = np.clip(norm, 0.0, 1.0)
        else:
            norm = np.zeros_like(depth_mm, dtype=np.float32)
        norm[~valid] = 0.0
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return vis


def render_pair(left: np.ndarray, right: np.ndarray, height: int = 360) -> np.ndarray:
    l = cv2.resize(left, (int(left.shape[1] * height / left.shape[0]), height))
    r = cv2.resize(right, (int(right.shape[1] * height / right.shape[0]), height))
    return cv2.hconcat([l, r])


def render_peaks_panel(
    warped_color: np.ndarray,
    peaks_warped: list[tuple[int, int, float]],
    world_xy_mm: list[tuple[float, float]],
    world_grid_points: np.ndarray | None = None,
    peak_squares: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    vis = warped_color.copy()
    h, w = vis.shape[:2]
    for i in range(1, 8):
        x = int(round(i * w / 8))
        y = int(round(i * h / 8))
        cv2.line(vis, (x, 0), (x, h - 1), (50, 50, 50), 1)
        cv2.line(vis, (0, y), (w - 1, y), (50, 50, 50), 1)

    for x, y, h in peaks_warped:
        cv2.circle(vis, (x, y), 6, (0, 0, 0), 2)
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"{h:.0f}mm",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    canvas = np.zeros((360, 360, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.line(canvas, (0, 180), (360, 180), (80, 80, 80), 1)
    cv2.line(canvas, (180, 0), (180, 360), (80, 80, 80), 1)
    cv2.putText(
        canvas,
        "World XY (mm)",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    max_range = 200.0
    if world_xy_mm:
        max_range = max(max_range, max(max(abs(x), abs(y)) for x, y in world_xy_mm))
    if world_grid_points is not None:
        max_range = max(
            max_range,
            float(np.max(np.abs(world_grid_points.reshape(-1, 2)))),
        )
    max_range *= 1.05

    def to_canvas(pt):
        x, y = pt
        cx = int(180 + (x / max_range) * 160)
        cy = int(180 - (y / max_range) * 160)
        return cx, cy

    if world_grid_points is not None:
        if peak_squares:
            overlay = canvas.copy()
            for r, c in peak_squares:
                quad = np.array(
                    [
                        world_grid_points[r, c],
                        world_grid_points[r, c + 1],
                        world_grid_points[r + 1, c + 1],
                        world_grid_points[r + 1, c],
                    ],
                    dtype=np.float32,
                )
                quad_canvas = np.array([to_canvas(pt) for pt in quad], dtype=np.int32)
                cv2.fillPoly(overlay, [quad_canvas], (60, 90, 160))
            canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

        for r in range(9):
            for c in range(8):
                p1 = to_canvas(world_grid_points[r, c])
                p2 = to_canvas(world_grid_points[r, c + 1])
                cv2.line(canvas, p1, p2, (70, 70, 70), 1)
        for c in range(9):
            for r in range(8):
                p1 = to_canvas(world_grid_points[r, c])
                p2 = to_canvas(world_grid_points[r + 1, c])
                cv2.line(canvas, p1, p2, (70, 70, 70), 1)

    for x, y in world_xy_mm:
        cx, cy = to_canvas((x, y))
        cv2.circle(canvas, (cx, cy), 4, (0, 255, 255), -1)

    return render_pair(vis, canvas, height=360)


def atomic_save(path: Path, array: np.ndarray) -> None:
    """Saves a NumPy array atomically."""
    tmp = path.with_suffix(".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, array)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def peaks_to_occupancy(peak_squares: set[tuple[int, int]]) -> np.ndarray:
    occupancy = np.zeros((8, 8), dtype=np.float32)
    for r, c in peak_squares:
        if 0 <= r < 8 and 0 <= c < 8:
            occupancy[r, c] = 1.0
    return occupancy


def main():
    parser = argparse.ArgumentParser(
        description="Detect height peaks and deproject to XYZ"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for occupied_bitmap.npy (default: .)",
    )
    parser.add_argument(
        "--write-interval",
        type=float,
        default=0.0,
        help="Seconds between occupancy writes (default: every frame)",
    )
    parser.add_argument(
        "--dont_visualize",
        action="store_true",
        help="Disable OpenCV visualization windows",
    )
    parser.add_argument(
        "--calibration",
        default="calibration.npz",
        help="Path to calibration file (default: calibration.npz)",
    )
    parser.add_argument(
        "--min-height", type=float, default=35.0, help="Minimum peak height in mm"
    )
    parser.add_argument(
        "--max-peaks", type=int, default=32, help="Maximum number of peaks to display"
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=15.0,
        help="Limit processing/visualization FPS (default: 15)",
    )
    parser.add_argument(
        "--occ-avg-frames",
        type=int,
        default=6,
        help="Occupancy moving average window in frames (default: 3, use 1 to disable)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied_path = out_dir / "occupied_bitmap.npy"

    print(f"Loading calibration from {args.calibration}...")
    try:
        mode, data, baseline, _ = load_calibration(args.calibration)
    except FileNotFoundError:
        print(f"ERROR: {args.calibration} not found. Run python3 calibrate.py first.")
        return

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

    cam = CameraStream()
    with cam:
        profile = cam.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()

        print("Camera ready. Press 'q' to stop.")
        min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
        last_write = 0.0
        occ_window = max(1, args.occ_avg_frames)
        occ_history: deque[np.ndarray] = deque(maxlen=occ_window)

        while True:
            frame_start = time.monotonic()
            color, depth = cam.get_frames()
            if color is None or depth is None:
                continue

            warped_color = warp_board(color, M, BOARD_PX, flags=cv2.INTER_LINEAR)
            warped_depth = warp_board(depth, M, BOARD_PX, flags=cv2.INTER_NEAREST)

            height_map = compute_height_map(warped_depth, baseline_map)
            peaks = find_peaks(
                height_map, min_height=args.min_height, max_peaks=args.max_peaks
            )

            cell = BOARD_PX / 8.0
            peak_squares_uv: set[tuple[int, int]] = set()
            for x, y, _ in peaks:
                r = int(y / cell) if cell > 0 else 0
                c = int(x / cell) if cell > 0 else 0
                r = max(0, min(7, r))
                c = max(0, min(7, c))
                peak_squares_uv.add((r, c))
            if peaks:
                pts = np.float32([[p[0], p[1]] for p in peaks]).reshape(-1, 1, 2)
                pts_raw = cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2)
            else:
                pts_raw = np.empty((0, 2), dtype=np.float32)

            world_xy_mm = []
            for u, v in pts_raw:
                ui, vi = int(round(u)), int(round(v))
                if ui < 0 or vi < 0 or ui >= depth.shape[1] or vi >= depth.shape[0]:
                    continue
                d_mm = sample_depth_mm(depth, u, v)
                if d_mm <= 0:
                    continue
                xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], d_mm / 1000.0)
                world_xy_mm.append((xyz[0] * 1000.0, -xyz[1] * 1000.0))

            world_grid_points = None
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

            peak_squares_world: set[tuple[int, int]] = set()
            if len(corner_world) == 4:
                tl, tr, br, bl = [np.array(p, dtype=np.float32) for p in corner_world]
                grid = np.zeros((9, 9, 2), dtype=np.float32)
                for r in range(9):
                    t = r / 8.0
                    left = tl + (bl - tl) * t
                    right = tr + (br - tr) * t
                    for c in range(9):
                        s = c / 8.0
                        grid[r, c] = left + (right - left) * s
                world_grid_points = grid

                uvec = (tr - tl)[:2]
                vvec = (bl - tl)[:2]
                A = np.array([[uvec[0], vvec[0]], [uvec[1], vvec[1]]], dtype=np.float32)
                det = float(np.linalg.det(A))
                if abs(det) > 1e-6:
                    A_inv = np.linalg.inv(A)
                    for xw, yw in world_xy_mm:
                        rel = np.array([xw - tl[0], yw - tl[1]], dtype=np.float32)
                        s, t = A_inv @ rel
                        if 0.0 <= s < 1.0 and 0.0 <= t < 1.0:
                            c = int(s * 8)
                            r = int(t * 8)
                            peak_squares_world.add((r, c))

            peak_squares = peak_squares_world if peak_squares_world else peak_squares_uv
            now = time.monotonic()
            if args.write_interval <= 0.0 or (now - last_write) >= args.write_interval:
                occupancy_raw = peaks_to_occupancy(peak_squares)
                occ_history.append(occupancy_raw)
                if len(occ_history) == 1:
                    occupancy = occupancy_raw
                else:
                    occupancy = np.mean(np.stack(occ_history, axis=0), axis=0)
                atomic_save(occupied_path, occupancy.astype(np.float32))
                last_write = now

            raw_panel = render_pair(color, colorize_depth(depth))
            warped_panel = render_pair(
                warped_color, colorize_depth(warped_depth), height=360
            )
            peaks_panel = render_peaks_panel(
                warped_color,
                peaks,
                world_xy_mm,
                world_grid_points,
                peak_squares,
            )

            if not args.dont_visualize:
                cv2.imshow("Raw (RGB | Depth)", raw_panel)
                cv2.imshow("Warped (RGB | Depth)", warped_panel)
                cv2.imshow("Peaks (Image | World XY)", peaks_panel)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if min_frame_time > 0:
                elapsed = time.monotonic() - frame_start
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
