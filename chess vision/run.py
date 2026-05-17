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
    CameraStream, BoardAnalyzer, IrregularBoardAnalyzer, load_calibration
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
        analyzer = IrregularBoardAnalyzer(data, baseline, threshold_mm=threshold)
        print("Mode: Irregular grid (per-square quad sampling)")
    else:
        analyzer = BoardAnalyzer(data, baseline, threshold_mm=threshold)
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

            while True:
                color, depth = cam.get_frames()
                if color is None or depth is None:
                    continue

                # Analyze every frame to keep the smoothing buffer updated (30 FPS)
                occupied, heights_full, threshold_mask, normalized_grid, piece_classes = analyzer.analyze(depth, color)

                now = time.monotonic()
                if args.once or (now - last_print) >= args.interval:
                    last_print = now

                    # Save outputs atomically
                    atomic_save(occupied_path, occupied)
                    atomic_save(height_path, normalized_grid)

                    # Terminal display
                    ts = time.strftime("%H:%M:%S")
                    n_pieces = occupied.sum()
                    print(f"[{ts}]  Pieces detected: {n_pieces}")
                    print_board(piece_classes)

                    if args.once:
                        print(f"Saved to: {out_dir}")
                        break

                # Optional live visualization windows (updates at camera FPS)
                if args.visualize:
                    vis = analyzer.visualize(color, occupied)
                    cv2.imshow("Chess Vision — Board State", vis)
                    
                    # 1. Full Resolution Height Map + Grid
                    heights_vis = cv2.normalize(heights_full, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heights_vis = cv2.applyColorMap(heights_vis, cv2.COLORMAP_JET)
                    
                    if hasattr(analyzer, 'M'):  # BoardAnalyzer (Warped space is 512x512)
                        for i in range(1, 8):
                            cv2.line(heights_vis, (i * 64, 0), (i * 64, 512), (255, 255, 255), 1)
                            cv2.line(heights_vis, (0, i * 64), (512, i * 64), (255, 255, 255), 1)
                    else:  # IrregularBoardAnalyzer (Image space is e.g. 1280x720)
                        from chess_vision import draw_grid
                        heights_vis = draw_grid(heights_vis, analyzer.grid, color=(255, 255, 255))
                        
                    cv2.imshow("Full Resolution Height Map", heights_vis)

                    # 2. Pixels passing threshold (Threshold Pass Mask)
                    if hasattr(analyzer, 'M'):
                        # Already 512x512
                        cv2.imshow("Pixels > 2cm (Threshold Pass)", threshold_mask)
                    else:
                        # 1280x720, let's resize to a smaller size for display (e.g. 640x360)
                        mask_resized = cv2.resize(threshold_mask, (640, 360), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("Pixels > 2cm (Threshold Pass)", mask_resized)

                    # 3. Normalized 8x8 Grid Map
                    grid_vis = cv2.resize(normalized_grid, (512, 512), interpolation=cv2.INTER_NEAREST)
                    grid_vis = (grid_vis * 255).astype(np.uint8)
                    grid_vis = cv2.applyColorMap(grid_vis, cv2.COLORMAP_JET)
                    
                    # Overlay numeric values on the 8x8 squares
                    for r in range(8):
                        for c in range(8):
                            val = normalized_grid[r, c]
                            text = f"{val:.2f}"
                            x = c * 64 + 12
                            y = r * 64 + 38
                            text_color = (255, 255, 255) if val < 0.5 else (0, 0, 0)
                            cv2.putText(grid_vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
                            
                    cv2.imshow("Normalized 8x8 Grid", grid_vis)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Window closed — stopping.")
                        break

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
