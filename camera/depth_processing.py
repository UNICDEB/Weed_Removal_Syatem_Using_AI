import pyrealsense2 as rs

def apply_depth_filters(depth_frame):

    # 1. Decimation
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)
    depth_frame = decimation.process(depth_frame)

    # 2. Depth to Disparity
    depth_to_disparity = rs.disparity_transform(True)
    depth_frame = depth_to_disparity.process(depth_frame)

    # 3. Spatial
    spatial = rs.spatial_filter()
    depth_frame = spatial.process(depth_frame)

    # 4. Temporal
    temporal = rs.temporal_filter()
    depth_frame = temporal.process(depth_frame)

    # 5. Disparity to Depth
    disparity_to_depth = rs.disparity_transform(False)
    depth_frame = disparity_to_depth.process(depth_frame)

    # 🔥 VERY IMPORTANT
    depth_frame = depth_frame.as_depth_frame()

    return depth_frame
