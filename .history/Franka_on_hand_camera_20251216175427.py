import mujoco
import numpy as np
import cv2
from MujocoSim import FR3Sim
import open3d as o3d
import os

# Enable offline rendering
os.environ["MUJOCO_GL"] = "osmesa"


def render_rgbd(renderer: mujoco.Renderer, data: mujoco.MjData, camera_id: int):
    """Captures RGB and Depth images from the specified MuJoCo camera."""
    renderer.update_scene(data, camera=camera_id)
    rgb_image = renderer.render()
    renderer.enable_depth_rendering()
    depth_image = renderer.render()
    renderer.disable_depth_rendering()
    return rgb_image, depth_image


def rgbd_to_pointcloud(model, data, camera_id, rgb_image, depth_image, width, height):
    """Converts RGB and Depth images to a colored point cloud in world coordinates."""
    fovy = model.cam_fovy[camera_id]
    f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    Fx, Fy = f, f
    Cx, Cy = width / 2.0, height / 2.0
    cam_pos = data.cam_xpos[camera_id]
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3)
    cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=False)
    Z = depth_image.copy()
    X = Z * (cc - Cx) / Fx
    Y = Z * (rr - Cy) / Fy
    points_camera = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    points_world = (cam_rot @ points_camera.T).T + cam_pos
    colors = rgb_image.reshape(-1, 3) / 255.0
    valid_points_mask = (Z.flatten() > model.vis.map.znear) & (
        Z.flatten() < model.vis.map.zfar
    )
    points_world = points_world[valid_points_mask]
    colors = colors[valid_points_mask]
    point_cloud = np.hstack([points_world, colors])
    return point_cloud


# Setup
my_robot = FR3Sim()
q, dq = my_robot.get_state()
# Camera setup
CAMERA_NAME = "hand_rgbd_cam"
camera_id = my_robot.model.cam(CAMERA_NAME).id
width, height = 640, 480
renderer = mujoco.Renderer(my_robot.model, height=height, width=width)
# Capture single frame
rgb_image, depth_image = render_rgbd(renderer, my_robot.data, camera_id)
point_cloud = rgbd_to_pointcloud(
    my_robot.model, my_robot.data, camera_id, rgb_image, depth_image, width, height
)
# Save images
cv2.imwrite("rgb_image.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(
    "depth_image.png",
    np.uint8(cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)),
)
# Save point cloud
np.save("point_cloud.npy", point_cloud)
print(f"Point Cloud size: {point_cloud.shape[0]} points")
print("Saved: rgb_image.png, depth_image.png, point_cloud.npy")

# Save as PLY file
try:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
    print("Saved: point_cloud.ply")
    
    # Optional: Visualize point cloud (comment out for fully headless operation)
    # o3d.visualization.draw_geometries([pcd])
except ImportError:
    print("Open3D not installed. Install with: pip install open3d")
