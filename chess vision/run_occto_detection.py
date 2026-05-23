"""
Chess Vision — Live Detection Loop
===================================
Reads calibration.npz, then continuously captures frames and writes
the board state as NumPy arrays that your chess engine can read.

Usage:
    python3 run.py                          # live loop, saves every second
    python3 run.py --once                   # single capture and exit
    python3 run.py --interval 0.5           # capture every 0.5 s
    python3 run.py --output-dir /tmp/chess  # write outputs to a custom dir
    python3 run.py --visualize              # show live annotated board window
    python3 run.py --max-fps 10             # limit processing / visualization FPS

Output files (written atomically via temp-file rename):
    <output_dir>/occupied_bitmap.npy   — shape (8,8) dtype bool
    <output_dir>/height_map_mm.npy     — shape (8,8) dtype float32

Array layout:
    Row 0 = rank 8 (back rank from white's perspective, nearest to ID0/ID1 markers)
    Row 7 = rank 1
    Col 0 = a-file, Col 7 = h-file
    (adjust in your chess engine if your board orientation differs)
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from chess_vision import (
    CameraStream, BoardAnalyzer, IrregularBoardAnalyzer, load_calibration,
    BOARD_PX, SQUARE_PX
)

RANK_LABELS = "87654321"
FILE_LABELS = "abcdefgh"


def print_board(piece_classes: np.ndarray) -> None:
    """Pretty-prints the 8×8 board state to stdout."""
    print(f"  {'  '.join(FILE_LABELS)}")
    print(f"  {'─' * 23}")
    for row in range(8):
        cells = "  ".join(str(piece_classes[row, c]) for c in range(8))
        print(f"{RANK_LABELS[row]} │ {cells}")
    print()


def atomic_save(path: Path, array: np.ndarray) -> None:
    """Saves a NumPy array atomically using a temp file + rename."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)  # atomic on Linux


def warp_board(image: np.ndarray, matrix: np.ndarray, size: int, flags: int) -> np.ndarray:
    return cv2.warpPerspective(image, matrix, (size, size), flags=flags)


def grid_corners(grid: np.ndarray) -> np.ndarray:
    return np.float32([grid[0, 0], grid[0, 8], grid[8, 8], grid[8, 0]])


def build_uniform_grid() -> np.ndarray:
    grid = np.zeros((9, 9, 2), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            grid[r, c] = (c * SQUARE_PX, r * SQUARE_PX)
    return grid


def render_height_overlay(
    color_bgr: np.ndarray,
    grid: np.ndarray,
    height_grid_mm: np.ndarray,
    max_height_mm: float = 80.0,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = color_bgr.copy()
    height_norm = np.clip(height_grid_mm / max_height_mm, 0.0, 1.0)
    for r in range(8):
        for c in range(8):
            quad = np.array([
                grid[r, c],
                grid[r, c + 1],
                grid[r + 1, c + 1],
                grid[r + 1, c],
            ], dtype=np.int32)
            color = cv2.applyColorMap(
                np.array([[int(height_norm[r, c] * 255)]], dtype=np.uint8),
                cv2.COLORMAP_JET
            )[0, 0].tolist()
            cv2.fillPoly(overlay, [quad], color)

    blended = cv2.addWeighted(overlay, alpha, color_bgr, 1 - alpha, 0)

    for r in range(8):
        for c in range(8):
            quad = np.array([
                grid[r, c],
                grid[r, c + 1],
                grid[r + 1, c + 1],
                grid[r + 1, c],
            ], dtype=np.int32)
            cv2.polylines(blended, [quad], isClosed=True, color=(40, 40, 40), thickness=1)
            cx, cy = quad.mean(axis=0).astype(int)
            text = f"{height_grid_mm[r, c]:.0f}mm"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(blended,
                          (cx - tw // 2 - 2, cy - th // 2 - 2),
                          (cx + tw // 2 + 2, cy + th // 2 + 2),
                          (0, 0, 0), -1)
            cv2.putText(blended, text, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return blended


def main():
    parser = argparse.ArgumentParser(description="Chess Vision live detection")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write occupied_bitmap.npy and height_map_mm.npy")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between captures (default: 1.0)")
    parser.add_argument("--once", action="store_true",
                        help="Capture one snapshot and exit")
    parser.add_argument("--visualize", action="store_false",
                        help="Show a live annotated board window (requires display)")
    parser.add_argument("--max-fps", type=float, default=15.0,
                        help="Limit processing/visualization FPS (default: 15)")
    parser.add_argument("--calibration", default="calibration.npz",
                        help="Path to calibration file (default: calibration.npz)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override occupancy threshold in mm (default: from calibration)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied_path = out_dir / "occupied_bitmap.npy"
    height_path = out_dir / "height_map_mm.npy"

    # ----------------------------------------------------------------
    # Load calibration
    # ----------------------------------------------------------------
    print(f"Loading calibration from {args.calibration}…")
    try:
        mode, data, baseline, threshold = load_calibration(args.calibration)
    except FileNotFoundError:
        print(f"ERROR: {args.calibration} not found.")
        print("Run  python3 calibrate.py  first to generate it.")
        return

    if args.threshold is not None:
        threshold = args.threshold
        print(f"Overriding occupancy threshold → {threshold} mm")

    if mode == "grid":
        grid_points = data.astype(np.float32)
        src = grid_corners(grid_points)
        dst = np.float32([
            [0, 0],
            [BOARD_PX, 0],
            [BOARD_PX, BOARD_PX],
            [0, BOARD_PX],
        ])
        warp_M = cv2.getPerspectiveTransform(src, dst)
        grid_warped = cv2.perspectiveTransform(
            grid_points.reshape(-1, 1, 2), warp_M
        ).reshape(9, 9, 2)
        analyzer = IrregularBoardAnalyzer(grid_warped, baseline, threshold_mm=threshold)
        overlay_grid = grid_warped
        print("Mode: Irregular grid (per-square quad sampling)")
    else:
        warp_M = data.astype(np.float32)
        analyzer = BoardAnalyzer(np.eye(3, dtype=np.float32), baseline, threshold_mm=threshold)
        overlay_grid = build_uniform_grid()
        print("Mode: 4-corner perspective transform")
    print(f"Occupancy threshold: {threshold} mm")
    print(f"Output directory:    {out_dir.resolve()}")
    print()

    # ----------------------------------------------------------------
    # Live capture loop
    # ----------------------------------------------------------------
    cam = CameraStream()
    with cam:
        print("Camera ready. Press Ctrl-C to stop.\n")
        try:
            last_print = 0.0
            min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0.0

            while True:
                frame_start = time.monotonic()
                color, depth = cam.get_frames()
                if color is None or depth is None:
                    continue

                warped_color = warp_board(color, warp_M, BOARD_PX, flags=cv2.INTER_LINEAR)
                warped_depth = warp_board(depth, warp_M, BOARD_PX, flags=cv2.INTER_NEAREST)

                # Analyze every frame to keep the smoothing buffer updated
                if mode == "grid":
                    occupied, heights_full, threshold_mask, normalized_grid, piece_classes, height_grid_mm = analyzer.analyze(warped_depth)
                else:
                    occupied, heights_full, threshold_mask, normalized_grid, piece_classes, height_grid_mm = analyzer.analyze(warped_depth, prewarped=True)

                now = time.monotonic()
                if args.once or (now - last_print) >= args.interval:
                    last_print = now

                    # Save outputs atomically
                    atomic_save(occupied_path, occupied)
                    atomic_save(height_path, height_grid_mm)

                    # Terminal display
                    ts = time.strftime("%H:%M:%S")
                    n_pieces = occupied.sum()
                    print(f"[{ts}]  Pieces detected: {n_pieces}")
                    print_board(piece_classes)

                    if args.once:
                        print(f"Saved to: {out_dir}")
                        break

                if args.visualize:
                    vis = render_height_overlay(warped_color, overlay_grid, height_grid_mm)
                    cv2.imshow("Chess Vision — Height Overlay", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Window closed — stopping.")
                        break

                if min_frame_time > 0:
                    elapsed = time.monotonic() - frame_start
                    if elapsed < min_frame_time:
                        time.sleep(min_frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
