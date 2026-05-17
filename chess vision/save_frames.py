import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

pipeline.start(config)

try:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Colorize depth for visualization
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )

    # Save files
    cv2.imwrite("color.png", color_image)
    cv2.imwrite("depth_colorized.png", depth_colormap)
    cv2.imwrite("depth_raw.png", depth_image)  # 16-bit raw depth in mm

    print("Saved color.png, depth_colorized.png, depth_raw.png")
finally:
    pipeline.stop()