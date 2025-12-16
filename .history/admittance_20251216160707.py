# FIX: The 'pyexpat' import seems incorrect for MuJoCo. Use 'mujoco' for its bindings.
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
from MujocoSim import FR3Sim


# import cv2 # Optional: uncomment if you need OpenCV to save or process images
# --- NEW: RGBD CAPTURE FUNCTION ---
def render_rgbd(renderer: mujoco.Renderer, data: mujoco.MjData, camera_id: int):
    """Captures RGB and Depth images from the specified MuJoCo camera."""
    # 1. Update the scene for the camera view (must be done before each render)
    # The 'data' object contains the current pose of the robot/camera
    renderer.update_scene(data, camera=camera_id)
    # 2. Render RGB (Color) Image
    rgb_image = renderer.render()
    # 3. Render Depth Image
    renderer.enable_depth_rendering()
    depth_image = renderer.render()  # Raw depth values in meters
    renderer.disable_depth_rendering()
    return rgb_image, depth_image


# -----------------------------------
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
# --- NEW: Camera Setup ---
CAMERA_NAME = "hand_rgbd_cam"
camera_id = my_robot.model.cam(CAMERA_NAME).id
width = my_robot.model.cam_width[camera_id]
height = my_robot.model.cam_height[camera_id]
# 4. NEW: Initialize the Renderer
renderer = mujoco.Renderer(my_robot.model, height=height, width=width)
# -------------------------
for i in range(steps):
    # ... (Your existing admittance control logic) ...
    q, dq = my_robot.get_state()
    T_current = my_robot.get_pose(q)
    x = T_current[:3, 3]
    J = my_robot.get_jacobian(q)
    R_ee = T_current[:3, :3]
    J_spatial = np.block(
        [[R_ee @ J[3:, :]], [R_ee @ J[:3, :]]]  # Linear part  # Angular part
    )
    dx = J_spatial @ dq
    dx = dx[:3]
    # Get external force from simulation (right-click + drag in Mujoco viewer)
    f_ext = my_robot.data.xfrc_applied[ee_body_id]
    # Admittance dynamics: M*ddx + D*dx + K*(x - x_des) = f_ext
    # Solve for ddx_des (use only force part)
    ddx_des = np.linalg.inv(M_adm) @ (f_ext[:3] - D_adm @ dx_des - K_adm @ (x_des - x))
    dx_des += ddx_des * dt
    x_des += dx_des * dt
    # Task-space impedance control to track admittance trajectory
    pos_error = x_des - x
    vel_error = dx_des - dx
    P = np.array([1500, 1500, 1500])
    D = np.array([300, 300, 300])
    desired_force = P * pos_error + D * vel_error
    # Compose desired wrench (force only, no orientation control here)
    desired_wrench = np.concatenate([desired_force, np.zeros(3)])
    tau = J_spatial.T @ desired_wrench + my_robot.get_gravity(q)
    my_robot.send_joint_torque(tau, 10)  # Keep gripper open
    # --- NEW: Capture RGBD at a lower frequency ---
    # Capturing images every step (0.001s) is usually too slow.
    # We capture every 100 steps (10 FPS) instead.
    if i % 100 == 0:
        rgb_image, depth_image = render_rgbd(renderer, my_robot.data, camera_id)
        # 'rgb_image' is a (H, W, 3) numpy array (uint8)
        # 'depth_image' is a (H, W) numpy array of float values (meters)
        # Example of how to use the data:
        print(
            f"Time: {my_robot.data.time:.3f}s, Captured RGBD frame. Max depth: {np.max(depth_image):.2f}m"
        )
        # To display or save, you would typically use an image library like OpenCV or Matplotlib.
    # ---------------------------------------------
    time.sleep(dt)
