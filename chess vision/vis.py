"""
Visualization script to demonstrate the tall piece bleeding problem and the fix.

This script:
1. Loads a depth frame and grid calibration
2. Shows occupancy detection BEFORE and AFTER the fix
3. Visualizes which pixels contribute to each square's occupancy decision
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_square_contamination(depth_mm, grid_points, baseline, square_r, square_c, shrink=0.3):
    """
    Shows what pixels are being sampled for a specific square.
    Highlights the problem: pixels from tall pieces outside the square get included.
    """
    h, w = depth_mm.shape[:2]
    
    # Define quad for this square
    quad_full = np.array([
        grid_points[square_r, square_c],
        grid_points[square_r, square_c + 1],
        grid_points[square_r + 1, square_c + 1],
        grid_points[square_r + 1, square_c],
    ], dtype=np.float32)
    
    # Shrink for occupancy
    centroid = quad_full.mean(axis=0)
    quad_shrunk = quad_full + (centroid - quad_full) * shrink
    quad_shrunk_int = quad_shrunk.astype(np.int32)
    quad_full_int = quad_full.astype(np.int32)
    
    # Create masks
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_full, [quad_full_int], 255)
    
    mask_shrunk = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_shrunk, [quad_shrunk_int], 255)
    
    # Calculate heights
    valid_full = (mask_full > 0) & (depth_mm > 0)
    heights = np.zeros_like(depth_mm, dtype=np.float32)
    if np.any(valid_full):
        heights[valid_full] = baseline[square_r, square_c] - depth_mm[valid_full]
    
    # Show visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Image 1: Full quad (BUGGY - includes stray pixels)
    img1 = np.zeros((h, w, 3), dtype=np.uint8)
    img1[mask_full > 0] = [255, 0, 0]  # Full region in red
    img1[mask_shrunk > 0] = [0, 255, 0]  # Shrunk region in green
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Sampling Regions for Square ({square_r}, {square_c})\nRed=Full Quad, Green=Shrunk")
    axes[0, 0].axis('off')
    
    # Image 2: Heights in this square (BUGGY)
    heights_display = heights.copy()
    heights_display[mask_full == 0] = 0
    im = axes[0, 1].imshow(heights_display, cmap='hot', vmin=0, vmax=50)
    axes[0, 1].set_title("Heights in Full Quad (includes stray pixels)")
    plt.colorbar(im, ax=axes[0, 1], label='Height (mm)')
    
    # Image 3: Heights above threshold in FULL quad (BUGGY)
    above_threshold_full = (heights > 20.0) & (mask_full > 0)
    img3 = np.zeros((h, w, 3), dtype=np.uint8)
    img3[above_threshold_full] = [0, 255, 0]  # Green where height > 20mm
    axes[1, 0].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    n_pixels_full = np.sum(above_threshold_full)
    axes[1, 0].set_title(f"Above Threshold in Full Quad (BUGGY)\nPixels detected: {n_pixels_full}")
    axes[1, 0].axis('off')
    
    # Image 4: Heights above threshold in SHRUNK quad (FIXED)
    above_threshold_shrunk = (heights > 20.0) & (mask_shrunk > 0)
    img4 = np.zeros((h, w, 3), dtype=np.uint8)
    img4[above_threshold_shrunk] = [0, 255, 0]  # Green where height > 20mm
    axes[1, 1].imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    n_pixels_shrunk = np.sum(above_threshold_shrunk)
    axes[1, 1].set_title(f"Above Threshold in Shrunk Quad (FIXED)\nPixels detected: {n_pixels_shrunk}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def compare_old_vs_new(depth_mm, grid_points, baseline, threshold_height=20.0, shrink=0.3):
    """
    Compare occupancy detection: old way (contaminated) vs new way (clean).
    """
    h, w = depth_mm.shape[:2]
    
    # OLD WAY: Build global heights_full, then sample from it
    heights_full_old = np.zeros_like(depth_mm, dtype=np.float32)
    for r in range(8):
        for c in range(8):
            quad = np.array([
                grid_points[r, c],
                grid_points[r, c + 1],
                grid_points[r + 1, c + 1],
                grid_points[r + 1, c],
            ], dtype=np.float32).astype(np.int32)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [quad], 255)
            valid = (mask > 0) & (depth_mm > 0)
            if np.any(valid):
                heights_full_old[valid] = baseline[r, c] - depth_mm[valid]
    
    occupancy_old = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            quad = np.array([
                grid_points[r, c],
                grid_points[r, c + 1],
                grid_points[r + 1, c + 1],
                grid_points[r + 1, c],
            ], dtype=np.float32)
            centroid = quad.mean(axis=0)
            quad_shrunk = quad + (centroid - quad) * shrink
            quad_shrunk_int = quad_shrunk.astype(np.int32)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [quad_shrunk_int], 255)
            valid = (mask > 0) & (depth_mm > 0)
            
            if np.any(valid):
                heights = heights_full_old[mask > 0]
                above = np.sum(heights > threshold_height)
                occupancy_old[r, c] = above / len(heights)
    
    # NEW WAY: Calculate heights LOCALLY per square
    occupancy_new = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            # Full quad
            quad_full = np.array([
                grid_points[r, c],
                grid_points[r, c + 1],
                grid_points[r + 1, c + 1],
                grid_points[r + 1, c],
            ], dtype=np.float32)
            
            # Shrunk quad
            centroid = quad_full.mean(axis=0)
            quad_shrunk = quad_full + (centroid - quad_full) * shrink
            quad_shrunk_int = quad_shrunk.astype(np.int32)
            
            mask_full = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_full, [quad_full.astype(np.int32)], 255)
            
            mask_shrunk = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_shrunk, [quad_shrunk_int], 255)
            
            # Calculate heights LOCALLY
            valid_full = (mask_full > 0) & (depth_mm > 0)
            if np.any(valid_full):
                heights_local = baseline[r, c] - depth_mm[valid_full]
                
                # Check occupancy in SHRUNK region only
                valid_shrunk = (mask_shrunk > 0) & (depth_mm > 0)
                if np.any(valid_shrunk):
                    heights_shrunk = baseline[r, c] - depth_mm[valid_shrunk]
                    above = np.sum(heights_shrunk > threshold_height)
                    occupancy_new[r, c] = above / len(heights_shrunk)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(occupancy_old, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[0].set_title("OLD (BUGGY): Global Height Map\n(Contaminated by adjacent pieces)")
    axes[0].set_xticks(range(8))
    axes[0].set_yticks(range(8))
    for i in range(8):
        for j in range(8):
            axes[0].text(j, i, f"{occupancy_old[i, j]:.2f}", ha='center', va='center', color='black', fontsize=9)
    plt.colorbar(im1, ax=axes[0], label='Occupancy Confidence')
    
    im2 = axes[1].imshow(occupancy_new, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title("NEW (FIXED): Local Height Calculation\n(No contamination)")
    axes[1].set_xticks(range(8))
    axes[1].set_yticks(range(8))
    for i in range(8):
        for j in range(8):
            axes[1].text(j, i, f"{occupancy_new[i, j]:.2f}", ha='center', va='center', color='black', fontsize=9)
    plt.colorbar(im2, ax=axes[1], label='Occupancy Confidence')
    
    plt.tight_layout()
    return fig, occupancy_old, occupancy_new


if __name__ == "__main__":
    print("Running quick visualization of the tall piece contamination issue and the fix...")
    
    # 1. Load calibration
    try:
        data = np.load('calibration.npz')
        grid_points = data['grid_points']
        baseline = data['baseline']
        print("Loaded calibration.npz successfully.")
    except Exception as e:
        print(f"Could not load calibration.npz: {e}. Generating mock calibration.")
        grid_points = np.zeros((9, 9, 2), dtype=np.float32)
        for r in range(9):
            for c in range(9):
                grid_points[r, c] = [100 + c * 60, 100 + r * 60]
        baseline = np.full((8, 8), 300.0, dtype=np.float32)

    # 2. Generate synthetic depth frame with a tall piece bleeding from (4, 4) into (3, 4)
    h, w = 720, 1280
    depth_mm = np.zeros((h, w), dtype=np.float32)
    
    # Fill background
    for r in range(8):
        for c in range(8):
            quad = np.array([
                grid_points[r, c],
                grid_points[r, c + 1],
                grid_points[r + 1, c + 1],
                grid_points[r + 1, c],
            ], dtype=np.float32).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [quad], 255)
            depth_mm[mask > 0] = baseline[r, c]

    # Center of square (4, 4)
    q44 = np.array([
        grid_points[4, 4],
        grid_points[4, 5],
        grid_points[5, 5],
        grid_points[5, 4],
    ], dtype=np.float32)
    center_44 = q44.mean(axis=0)
    
    # Center of square (3, 4)
    q34 = np.array([
        grid_points[3, 4],
        grid_points[3, 5],
        grid_points[4, 5],
        grid_points[4, 4],
    ], dtype=np.float32)
    center_34 = q34.mean(axis=0)
    
    # Place a tall piece base in (4, 4) with depth = baseline - 80.0
    cv2.circle(depth_mm, tuple(center_44.astype(int)), 18, float(baseline[4, 4] - 80.0), -1)
    
    # Project/lean the top of the piece 50% of the way into square (3, 4)
    top_center = center_44 + (center_34 - center_44) * 0.5
    cv2.line(depth_mm, tuple(center_44.astype(int)), tuple(top_center.astype(int)), float(baseline[4, 4] - 80.0), 24)
    cv2.circle(depth_mm, tuple(top_center.astype(int)), 12, float(baseline[4, 4] - 80.0), -1)

    # 3. Compare old vs new
    fig, occ_old, occ_new = compare_old_vs_new(depth_mm, grid_points, baseline)
    
    # Save the figure
    output_path = Path("tall_piece_comparison.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path.resolve()}")
    
    # Show contamination detail for square (3, 4)
    fig_detail = visualize_square_contamination(depth_mm, grid_points, baseline, 3, 4)
    detail_path = Path("square_contamination_detail.png")
    fig_detail.savefig(detail_path, dpi=150)
    print(f"Saved square contamination detail to {detail_path.resolve()}")
    
    plt.show()