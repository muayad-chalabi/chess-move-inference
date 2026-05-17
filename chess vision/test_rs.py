import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    print(f"Resetting {dev.get_info(rs.camera_info.name)}...")
    dev.hardware_reset()
