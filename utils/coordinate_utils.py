import pyrealsense2 as rs
import numpy as np

def pixel_to_3d(depth_frame, intrinsics, x, y):

    depth = depth_frame.get_distance(x, y)

    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)

    return depth, point  # depth, (X,Y,Z)
