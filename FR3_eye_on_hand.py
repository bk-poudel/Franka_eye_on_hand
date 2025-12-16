import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2  # NEW: Import OpenCV for window display

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
    # 1. Get Intrinsic Parameters
    fovy = model.cam_fovy[camera_id]
    f = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    Fx, Fy = f, f
    Cx, Cy = width / 2.0, height / 2.0
    # 2. Get Extrinsic Parameters
    cam_pos = data.cam_xpos[camera_id]
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3)
    # 3. Create grid of pixel coordinates
    cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=False)
    # 4. Unproject to 3D camera coordinates
    Z = depth_image.copy()
    X = Z * (cc - Cx) / Fx
    Y = Z * (rr - Cy) / Fy
    points_camera = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # 5. Transform 3D points to World frame
    points_world = (cam_rot @ points_camera.T).T + cam_pos
    # 6. Prepare Colors
    colors = rgb_image.reshape(-1, 3) / 255.0
    # Filter out invalid depth points (optional but recommended)
    valid_points_mask = (Z.flatten() > model.vis.map.znear) & (
        Z.flatten() < model.vis.map.zfar
    )
    points_world = points_world[valid_points_mask]
    colors = colors[valid_points_mask]
    point_cloud = np.hstack([points_world, colors])
    return point_cloud


# --- Main Simulation Setup ---
my_robot = FR3Sim()
# Admittance parameters (virtual mass, damping, stiffness)
M_adm = np.diag([2.0, 2.0, 2.0])
D_adm = np.diag([300.0, 300.0, 300.0])
K_adm = np.diag([2000.0, 2000.0, 2000.0])
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
    exit()
# Fix for the AttributeError: Attempt to get resolution, with a fallback
try:
    width = my_robot.model.cam_width[camera_id]
    height = my_robot.model.cam_height[camera_id]
except AttributeError:
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
    # --- Capture and Display RGBD ---
    if i % 100 == 0:
        rgb_image, depth_image = render_rgbd(renderer, my_robot.data, camera_id)
        # 1. Display RGB image
        # MuJoCo output is RGB, but OpenCV expects BGR, so we convert it.
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Camera RGB", rgb_bgr)
        # 2. Display Depth image (Normalized for visualization)
        # Normalize the raw float depth values (meters) to 0-255 for display
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_display = np.uint8(depth_normalized)
        cv2.imshow("Hand Camera Depth (Normalized)", depth_display)
        # Update the windows
        # WaitKey(1) allows the windows to refresh and keeps the simulation responsive.
        cv2.waitKey(1)
        # Point Cloud conversion (optional, but good for data processing)
        point_cloud = rgbd_to_pointcloud(
            my_robot.model,
            my_robot.data,
            camera_id,
            rgb_image,
            depth_image,
            width,
            height,
        )
        print(
            f"Time: {my_robot.data.time:.3f}s | Point Cloud size: {point_cloud.shape[0]} points."
        )
    # --------------------------------
    time.sleep(dt)
# Clean up windows when the script finishes or is interrupted
cv2.destroyAllWindows()
