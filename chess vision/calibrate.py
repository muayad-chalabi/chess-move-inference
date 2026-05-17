"""
Calibration — snap-click mode.

Usage:  python3 calibrate.py

UI:
  • Camera captures a frame.
  • Detected corners shown as cyan dots.
  • LEFT-CLICK  near a corner → snaps to it (green dot).
  • RIGHT-CLICK → undo last point.
  • Close the window when done.
  The system auto-organizes your clicks into a 9×9 grid.
"""

import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from chess_vision import (
    CameraStream, DEFAULT_THRESHOLD_MM,
    draw_grid, save_calibration_grid,
)

N_BASELINE_FRAMES = 40
SNAP_RADIUS = 30          # px — snap to corner if this close


# ── Camera ──────────────────────────────────────────────────────────────────

def grab_frame(cam, warmup=15):
    print(f"  Warming up ({warmup} frames)…", end="", flush=True)
    color = depth = None
    for _ in range(warmup):
        color, depth = cam.get_frames()
        print(".", end="", flush=True)
    print(" done.")
    return color, depth


# ── Corner detection ─────────────────────────────────────────────────────────

def detect_corners(color_bgr, max_corners=600):
    """Return (N,2) float32 array of Harris/Shi-Tomasi corners."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(
        np.float32(gray), maxCorners=max_corners,
        qualityLevel=0.01, minDistance=12)
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2)


def snap(x, y, corners):
    """Snap (x,y) to nearest detected corner if within SNAP_RADIUS."""
    if len(corners) == 0:
        return x, y
    d = np.hypot(corners[:, 0] - x, corners[:, 1] - y)
    idx = np.argmin(d)
    if d[idx] < SNAP_RADIUS:
        return float(corners[idx, 0]), float(corners[idx, 1])
    return x, y


# ── Grid fitting ─────────────────────────────────────────────────────────────

def _board_corners(pts):
    """Pick TL, TR, BR, BL from a point set using convex hull."""
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmax(hull[:, 0] - hull[:, 1])]
    br = hull[np.argmax(hull[:, 0] + hull[:, 1])]
    bl = hull[np.argmin(hull[:, 0] - hull[:, 1])]
    return tl, tr, br, bl


def fit_grid(clicked):
    """
    Build a (9,9,2) float32 grid from N clicked points.

    Algorithm:
      1. Estimate board corners from convex hull of clicked points.
      2. Compute perspective transform  clicked-coords → unit [0,8]×[0,8] grid.
      3. Project each clicked point to get its approximate (row, col).
      4. Round to nearest integer grid position and store.
      5. Fill any missing positions by linear interpolation along rows/cols.
    """
    pts = np.array(clicked, dtype=np.float32)

    if len(pts) < 4:
        raise ValueError(f"Need at least 4 points, got {len(pts)}.")

    # Step 1-2: perspective from clicked corners to 0..8 grid space
    tl, tr, br, bl = _board_corners(pts)
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
    M = cv2.getPerspectiveTransform(src, dst)

    # Step 3: project all points to grid space
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    proj = (M @ pts_h.T).T
    grid_xy = proj[:, :2] / proj[:, 2:3]   # (N, 2) in 0..8 range

    # Step 4: assign to integer (row, col) positions
    sparse = {}      # (row, col) → list of original image-pts
    for img_pt, g in zip(pts, grid_xy):
        r = int(np.round(np.clip(g[1], 0, 8)))
        c = int(np.round(np.clip(g[0], 0, 8)))
        sparse.setdefault((r, c), []).append(img_pt)

    grid = np.full((9, 9, 2), np.nan, dtype=np.float32)
    for (r, c), pts_list in sparse.items():
        grid[r, c] = np.mean(pts_list, axis=0)

    # Step 5: fill missing by bilinear interpolation from board corners
    M_inv = cv2.getPerspectiveTransform(dst, src)
    for r in range(9):
        for c in range(9):
            if np.isnan(grid[r, c, 0]):
                pt_grid = np.float32([[[c, r]]])
                pt_img = cv2.perspectiveTransform(pt_grid, M_inv)
                grid[r, c] = pt_img[0, 0]

    return grid


# ── Matplotlib snap-click UI ──────────────────────────────────────────────────

def snap_click_ui(color_bgr, detected_corners):
    """
    Show image, let user click (snapping to detected corners).
    Returns list of (x, y) clicked points.
    """
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    clicked = []
    artists = []

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(rgb)
    ax.axis("off")

    # Show detected corners
    if len(detected_corners):
        ax.scatter(detected_corners[:, 0], detected_corners[:, 1],
                   s=12, c="cyan", alpha=0.5, zorder=3, label="Detected corners")

    ax.set_title(
        "Click on grid intersections  (snaps to cyan dots)\n"
        "LEFT-CLICK = add  │  RIGHT-CLICK = undo  │  Close when done",
        fontsize=11, pad=8)
    ax.legend(loc="lower left", fontsize=8)

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:          # add
            sx, sy = snap(event.xdata, event.ydata, detected_corners)
            clicked.append((sx, sy))
            sc = ax.scatter(sx, sy, s=70, c="lime",
                            edgecolors="black", linewidths=0.8, zorder=6)
            tx = ax.text(sx + 5, sy - 7, str(len(clicked)),
                         fontsize=6, color="white", zorder=7)
            artists.append((sc, tx))
            ax.set_title(f"{len(clicked)} points marked — keep clicking, or close when done",
                         fontsize=11)
            fig.canvas.draw()
        elif event.button == 3 and clicked:   # undo
            clicked.pop()
            sc, tx = artists.pop()
            sc.remove(); tx.remove()
            ax.set_title(f"{len(clicked)} points marked — keep clicking, or close when done",
                         fontsize=11)
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
    return clicked


def confirm_grid_ui(color_bgr, grid):
    """Show the fitted grid. Returns True if user confirms (closes window)."""
    annotated = draw_grid(color_bgr, grid)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("Detected grid — close to confirm, or re-run calibrate.py to redo.", fontsize=11)
    plt.tight_layout()
    plt.show()


# ── Depth baseline ────────────────────────────────────────────────────────────

def capture_baseline(cam, grid):
    """Capture N_BASELINE_FRAMES frames, return baseline and adaptive threshold maps."""
    from chess_vision import IrregularBoardAnalyzer
    shrink = IrregularBoardAnalyzer.SHRINK

    def shrink_quad(pts):
        c = pts.mean(axis=0)
        return (pts + (c - pts) * shrink).astype(np.int32)

    history = np.zeros((N_BASELINE_FRAMES, 8, 8), dtype=np.float32)

    print(f"Capturing {N_BASELINE_FRAMES} baseline frames…")
    n = 0
    while n < N_BASELINE_FRAMES:
        color, depth = cam.get_frames()
        if color is None:
            continue
        h, w = depth.shape[:2]
        for r in range(8):
            for c in range(8):
                quad = np.array([grid[r, c], grid[r, c+1],
                                 grid[r+1, c+1], grid[r+1, c]], dtype=np.float32)
                inner = shrink_quad(quad)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [inner], 255)
                valid = depth[mask > 0]
                valid = valid[valid > 0]
                if len(valid):
                    history[n, r, c] = float(np.median(valid))
        n += 1
        print(f"  {n}/{N_BASELINE_FRAMES}", end="\r", flush=True)
        time.sleep(0.05)
    print()
    
    baseline = np.zeros((8, 8), dtype=np.float32)
    threshold = np.zeros((8, 8), dtype=np.float32)
    
    MU_NOISE = 10.0
    K = 3.0
    
    for r in range(8):
        for c in range(8):
            vals = history[:, r, c]
            vals = vals[vals > 0]
            if len(vals) > 0:
                baseline[r, c] = np.median(vals)
                sigma = np.std(vals)
                threshold[r, c] = MU_NOISE + K * sigma
            else:
                threshold[r, c] = MU_NOISE

    return baseline, threshold


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Chess Vision — Calibration")
    print("=" * 50)

    cam = CameraStream()
    with cam:
        print("\nStep 1 — Grabbing frame…")
        color, depth = grab_frame(cam)
        if color is None:
            print("ERROR: No frame from camera.")
            return

        print("Step 2 — Detecting corners in image…")
        corners = detect_corners(color)
        print(f"  Found {len(corners)} candidate corners.")

        print("Step 3 — Mark your grid intersections.")
        print("  Cyan dots = auto-detected corners (clicks snap to these).")
        print("  You don't need to click ALL 81 — aim for ~20-40 spread evenly.")
        print("  The system fills in the rest automatically.\n")
        input("  Press ENTER to open the image… ")

        while True:
            clicked = snap_click_ui(color, corners)
            if len(clicked) < 4:
                print(f"  Only {len(clicked)} points — need at least 4. Please try again.")
                continue
            print(f"  {len(clicked)} points clicked. Fitting grid…")
            try:
                grid = fit_grid(clicked)
                confirm_grid_ui(color, grid)
                break
            except Exception as e:
                print(f"  Grid fitting failed: {e}. Please try again.")

        print("\nStep 4 — Remove ALL pieces from the board.")
        input("Press ENTER when board is empty… ")
        baseline, threshold = capture_baseline(cam, grid)

    print("\nBaseline depth (mm):")
    print(f"  [0,0]={baseline[0,0]:.1f}  [0,7]={baseline[0,7]:.1f}")
    print(f"  [7,0]={baseline[7,0]:.1f}  [7,7]={baseline[7,7]:.1f}")
    
    print("\nAdaptive Threshold map (mm):")
    print(f"  [0,0]={threshold[0,0]:.1f}  [0,7]={threshold[0,7]:.1f}")
    print(f"  [7,0]={threshold[7,0]:.1f}  [7,7]={threshold[7,7]:.1f}")

    save_calibration_grid(grid, baseline, threshold=threshold)
    print("\n✓ Done! Run  python3 run.py  to start detection.")


if __name__ == "__main__":
    main()
