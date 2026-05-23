"""
Chess Vision — Live Detection Loop (v2 with Diagnostics)
=========================================================

Displays per-square diagnostics:
  - Max height (mm)
  - % of square above height threshold
  
Tunable thresholds:
  --height-threshold: Height to count as occupied (default 20mm)
  --percentage-threshold: % of square that must be above height (default 15%)
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from chess_vision import (
    CameraStream, IrregularBoardAnalyzer, load_calibration,
    BOARD_PX, SQUARE_PX
)

RANK_LABELS = "87654321"
FILE_LABELS = "abcdefgh"


def print_board(piece_classes: np.ndarray) -> None:
    """Pretty-prints the 8×8 board state."""
    print(f"  {'  '.join(FILE_LABELS)}")
    print(f"  {'-' * 23}")
    for row in range(8):
        cells = "  ".join(str(piece_classes[row, c]) for c in range(8))
        print(f"{RANK_LABELS[row]} | {cells}")
    print()


def print_diagnostics(diagnostics: list, occupied_bitmap: np.ndarray) -> None:
    """Print detailed per-square diagnostics."""
    print("\nDetailed Per-Square Diagnostics:")
    print("=" * 100)
    print(f"{'Sq':<4} {'Max H':<8} {'% Above':<10} {'Pixels':<12} {'Occ':<4}")
    print("-" * 100)
    
    for r in range(8):
        for c in range(8):
            diag = diagnostics[r][c]
            max_h = diag['max_height_mm']
            pct = diag['percentage_above_threshold']
            pixels = f"{diag['num_pixels_above']}/{diag['total_pixels']}"
            occ = "Y" if occupied_bitmap[r, c] else "."
            
            sq_name = f"{FILE_LABELS[c]}{RANK_LABELS[r]}"
            print(f"{sq_name:<4} {max_h:>6.1f}mm {pct:>8.1%} {pixels:>12} {occ:>4}")
    
    print("=" * 100)


def atomic_save(path: Path, array: np.ndarray) -> None:
    """Saves a NumPy array atomically."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def render_diagnostics_overlay(
    color_bgr: np.ndarray,
    grid: np.ndarray,
    diagnostics: list,
    occupied_bitmap: np.ndarray,
) -> np.ndarray:
    """
    Overlay per-square diagnostics on the image:
      - Max height (top)
      - % above threshold (bottom)
      - Color coding by occupancy
    """
    vis = color_bgr.copy()
    
    for r in range(8):
        for c in range(8):
            quad = np.array([
                grid[r, c],
                grid[r, c + 1],
                grid[r + 1, c + 1],
                grid[r + 1, c],
            ], dtype=np.int32)
            
            # Color based on occupancy
            occ_color = (0, 200, 0) if occupied_bitmap[r, c] else (100, 100, 200)
            cv2.polylines(vis, [quad], isClosed=True, color=occ_color, thickness=2)
            
            # Get diagnostics
            diag = diagnostics[r][c]
            max_h = diag['max_height_mm']
            pct = diag['percentage_above_threshold']
            
            # Draw text in center of square
            centroid = quad.mean(axis=0).astype(int)
            
            # Top line: max height
            text_h = f"{max_h:.0f}mm"
            (tw, th), _ = cv2.getTextSize(text_h, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis,
                         (centroid[0] - tw // 2 - 2, centroid[1] - 12 - th),
                         (centroid[0] + tw // 2 + 2, centroid[1] - 12),
                         (0, 0, 0), -1)
            cv2.putText(vis, text_h, (centroid[0] - tw // 2, centroid[1] - 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Bottom line: percentage
            text_p = f"{pct:.0%}"
            (tw, th), _ = cv2.getTextSize(text_p, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis,
                         (centroid[0] - tw // 2 - 2, centroid[1] + 12),
                         (centroid[0] + tw // 2 + 2, centroid[1] + 12 + th),
                         (0, 0, 0), -1)
            cv2.putText(vis, text_p, (centroid[0] - tw // 2, centroid[1] + 12 + th),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return vis


def main():
    parser = argparse.ArgumentParser(description="Chess Vision with dual-threshold detection")
    parser.add_argument("--output-dir", default=".",
                       help="Output directory")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Seconds between captures")
    parser.add_argument("--once", action="store_true",
                       help="Single capture and exit")
    parser.add_argument("--dont_visualize", action="store_true",
                       help="Show live annotated board")
    parser.add_argument("--max-fps", type=float, default=15.0,
                       help="Max FPS")
    parser.add_argument("--calibration", default="calibration.npz",
                       help="Calibration file")
    parser.add_argument("--height-threshold", type=float, default=10.0,
                       help="Height threshold in mm (default 20)")
    parser.add_argument("--percentage-threshold", type=float, default=0.4,
                       help="Percentage threshold 0-1 (default 0.15 = 15%)")
    parser.add_argument("--diagnostics", action="store_true",
                       help="Print detailed per-square diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied_path = out_dir / "occupied_bitmap.npy"
    height_path = out_dir / "height_map_mm.npy"

    # Load calibration
    print(f"Loading calibration from {args.calibration}…")
    try:
        mode, data, baseline, threshold = load_calibration(args.calibration)
    except FileNotFoundError:
        print(f"ERROR: {args.calibration} not found.")
        return

    if mode != "grid":
        print("ERROR: This version requires 'grid' mode (irregular board).")
        return

    grid_points = data.astype(np.float32)
    
    # Create analyzer with tunable thresholds
    analyzer = IrregularBoardAnalyzer(
        grid_points,
        baseline,
        height_threshold_mm=args.height_threshold,
        occupancy_percentage_threshold=args.percentage_threshold,
    )
    
    print(f"\n{'='*60}")
    print("Chess Vision - Live Detection")
    print(f"{'='*60}")
    print(f"Height threshold:      {args.height_threshold} mm")
    print(f"Percentage threshold:  {args.percentage_threshold:.1%}")
    print(f"Output directory:      {out_dir.resolve()}")
    print(f"{'='*60}\n")

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

                # Analyze
                occupied, heights_full, threshold_mask, normalized_grid, piece_classes, height_grid_mm, diagnostics = \
                    analyzer.analyze(depth, color_bgr=color)

                now = time.monotonic()
                if args.once or (now - last_print) >= args.interval:
                    last_print = now

                    # Save
                    atomic_save(occupied_path, occupied)
                    atomic_save(height_path, height_grid_mm)

                    # Display
                    ts = time.strftime("%H:%M:%S")
                    n_pieces = occupied.sum()
                    print(f"[{ts}]  Pieces detected: {n_pieces}")
                    print_board(piece_classes)
                    
                    if args.diagnostics:
                        print_diagnostics(diagnostics, occupied)

                    if args.once:
                        print(f"Saved to: {out_dir}")
                        break

                if not args.dont_visualize:
                    vis = render_diagnostics_overlay(color, grid_points, diagnostics, occupied)
                    cv2.imshow("Chess Vision — Diagnostics", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Stopped.")
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