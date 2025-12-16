import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time

# You will likely need Open3D (pip install open3d) for visualization
# import open3d as o3d
# Assuming this class handles your MuJoCo environment setup
from MujocoSim import FR3Sim


# --- Helper Functions for RGBD Camera ---
def render_rgbd(renderer: mujoco.Renderer, data: mujoco.MjData, camera_id: int):
    """Captures RGB and Depth images from the specified MuJoCo camera."""
    # Update the scene for the specified camera
    renderer.update_scene(data, camera=camera_id)
    # 1. Render RGB (Color) Image
    rgb_image = renderer.render()
    # 2. Render Depth Image
    renderer.enable_depth_rendering()
    depth_image = renderer.render()  # Raw depth values in meters
    renderer.disable_depth_rendering()
    return rgb_image, depth_image


def rgbd_to_pointcloud(model, data, camera_id, rgb_image, depth_image, width, height):
    """Converts RGB and Depth images to a colored point cloud in world coordinates."""
    # 1. Get Intrinsic Parameters (Focal Lengths Fx, Fy, and Principal Points Cx, Cy)
    fovy = model.cam_fovy[camera_id]
    # Calculate focal length based on vertical FOV and height (assuming ideal pinhole)
    f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    Fx, Fy = f, f
    Cx, Cy = width / 2.0, height / 2.0
    # 2. Get Extrinsic Parameters (Camera Pose in World)
    cam_pos = data.cam_xpos[camera_id]
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3)
    # 3. Create grid of pixel coordinates (for unprojection)
    cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=False)
    # 4. Unproject (2D pixel + depth) to 3D camera coordinates (X, Y, Z)
    Z = depth_image.copy()
    X = Z * (cc - Cx) / Fx
    Y = Z * (rr - Cy) / Fy
    # points_camera shape: (H*W, 3)
    points_camera = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # 5. Transform 3D points from Camera frame to World frame
    # R * P_cam + T
    points_world = (cam_rot @ points_camera.T).T + cam_pos
    # 6. Prepare Colors
    colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize colors to [0, 1]
    # Optional: Filter out points at the far clip (MuJoCo's background/invalid depth)
    # The depth values are clamped between z_near and z_far.
    valid_points_mask = (Z.flatten() > model.vis.map.znear) & (
        Z.flatten() < model.vis.map.zfar
    )
    points_world = points_world[valid_points_mask]
    colors = colors[valid_points_mask]
    # Combined Point Cloud: (N, 6) -> [X, Y, Z, R, G, B]
    point_cloud = np.hstack([points_world, colors])
    return point_cloud


# --- Main Simulation Setup ---
my_robot = FR3Sim()
# Admittance parameters (virtual mass, damping, stiffness)
M_adm = np.diag([2.0, 2.0, 2.0])  # Virtual mass (kg)
D_adm = np.diag([300.0, 300.0, 300.0])  # Virtual damping (N·s/m)
K_adm = np.diag([2000.0, 2000.0, 2000.0])  # Virtual stiffness (N/m)
# Initial desired position is current end-effector position
q, dq = my_robot.get_state()
T_current = my_robot.get_pose(q)
x_des = T_current[:3, 3].copy()
dx_des = np.zeros(3)
dt = 0.001
steps = 50000
ee_body_id = my_robot.model.body(b"hand").id
# --- Camera Initialization ---
CAMERA_NAME = "hand_rgbd_cam"
try:
    camera_id = my_robot.model.cam(CAMERA_NAME).id
except ValueError:
    print(
        f"Error: Camera '{CAMERA_NAME}' not found in the model. Please check your XML."
    )
    # Exit or raise error if camera is critical
    exit()
# Fix for the AttributeError: Attempt to get resolution, with a fallback
try:
    # Standard MuJoCo binding access
    width = my_robot.model.cam_width[camera_id]
    height = my_robot.model.cam_height[camera_id]
except AttributeError:
    # Fallback to the resolution defined in the XML (640x480)
    print(
        "Warning: Failed to access model.cam_width/height. Using XML defined fallback resolution: 640x480."
    )
    width = 640
    height = 480
# Initialize the Renderer
renderer = mujoco.Renderer(my_robot.model, height=height, width=width)
# -----------------------------
for i in range(steps):
    # Get current state and kinematics
    q, dq = my_robot.get_state()
    T_current = my_robot.get_pose(q)
    x = T_current[:3, 3]
    J = my_robot.get_jacobian(q)
    R_ee = T_current[:3, :3]
    J_spatial = np.block([[R_ee @ J[3:, :]], [R_ee @ J[:3, :]]])
    dx = J_spatial @ dq
    dx = dx[:3]
    # Admittance Control Logic
    f_ext = my_robot.data.xfrc_applied[ee_body_id]
    ddx_des = np.linalg.inv(M_adm) @ (f_ext[:3] - D_adm @ dx_des - K_adm @ (x_des - x))
    dx_des += ddx_des * dt
    x_des += dx_des * dt
    # Task-space impedance control
    pos_error = x_des - x
    vel_error = dx_des - dx
    P = np.array([1500, 1500, 1500])
    D = np.array([300, 300, 300])
    desired_force = P * pos_error + D * vel_error
    desired_wrench = np.concatenate([desired_force, np.zeros(3)])
    tau = J_spatial.T @ desired_wrench + my_robot.get_gravity(q)
    my_robot.send_joint_torque(tau, 10)
    # --- Capture and Process RGBD ---
    # Capture images every 100 steps (e.g., 10 Hz if dt=0.001)
    if i % 100 == 0:
        rgb_image, depth_image = render_rgbd(renderer, my_robot.data, camera_id)
        # Convert to colored point cloud
        point_cloud = rgbd_to_pointcloud(
            my_robot.model,
            my_robot.data,
            camera_id,
            rgb_image,
            depth_image,
            width,
            height,
        )
        # 'point_cloud' is a NumPy array of shape (N, 6)
        # Columns: [World_X, World_Y, World_Z, Normalized_R, Normalized_G, Normalized_B]
        print(
            f"Time: {my_robot.data.time:.3f}s | Point Cloud size: {point_cloud.shape[0]} points."
        )
        #
        # --- Point Cloud Usage Example (Requires Open3D) ---
        # if i == 1000:
        #     # Create an Open3D point cloud object
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        #     pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
        #     o3d.io.write_point_cloud("hand_view_frame_1000.ply", pcd)
        #     print("Saved point cloud to hand_view_frame_1000.ply")
    # --------------------
    time.sleep(dt)
