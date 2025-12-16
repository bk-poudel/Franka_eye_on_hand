import numpy as np
import matplotlib.pyplot as plt
import time
from MujocoSim import FR3Sim

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
for i in range(steps):
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
    time.sleep(dt)
