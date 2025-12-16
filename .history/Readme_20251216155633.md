# Admittance_Control_FR3
Lightweight README for the Admittance Control project targeting the FR3 robot platform. Includes controller code, simulation helpers, example scripts, and test utilities.
## Table of contents
- Overview
- Features
- Requirements
- Quick start
- Example usage
- Configuration
- Testing
- Contributing
- License & contact
## Overview
Admittance_Control_FR3 implements an admittance-style force/impedance controller for the FR3 robot. The repo contains controller code (C++ and Python), configuration files, example launch scripts for simulation, and tests for validation on both simulation and hardware.
## Features
- Admittance controller core (C++ and Python) — modular and real-time friendly
- Configurable mass, damping, stiffness parameters
- Simulation support (MuJoCo examples / stubs for Gazebo)
- Logging and replay utilities
- Unit and integration tests
- ROS 2 publisher/subscriber hooks (optional)
## Requirements
Pip-installable:
- numpy
- scipy
- matplotlib
- mujoco
- pinocchio
System / environment:
- Linux (Ubuntu recommended)
- Python 3.8+ (virtualenv recommended)
- MuJoCo and Python bindings (see https://mujoco.org)
- Pinocchio (conda-forge or build from source)
- Optional: ROS 2 (Humble/Foxy or your distro) for rclpy and message packages
Example requirements.txt contents:
```
numpy
scipy
matplotlib
mujoco
pinocchio
```
## Quick start
1. Clone the repo:
```
git clone https://github.com/your-org/Admittance_Control_FR3.git
cd Admittance_Control_FR3
```
2. Python install:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pip install -e .
```
3. Run examples (see Example usage).
If building C++ components:
```
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
## Example usage
This repository includes two runnable Python examples:
- MujocoSim.py — simulation helper that provides the FR3Sim class.
    - Example instantiation:
    ```python
    from MujocoSim import FR3Sim
    sim = FR3Sim(interface_type="torque", render=True, dt=0.001, xml_path=None)
    ```
    - Run the default simulation (viewer):
    ```
    python MujocoSim.py
    ```
- admittance.py — example admittance controller that uses FR3Sim to read state and send joint torques:
```
python admittance.py
```
Notes:
- Example scripts are simple runners and do not implement a full CLI. Import classes programmatically for custom runs.
## Running on hardware
- Ensure correct URDF, joint mappings, and sign conventions.
- Use conservative gains initially and enable safety limits.
- If using ROS 2: source your ROS workspace and ensure required message packages are available.
## Troubleshooting
- Controller unstable: check force sign conventions, reduce gains, enable logging and replay.
- Real-time issues: use a real-time kernel or set proper thread priorities; reduce loop jitter.
- MuJoCo errors: confirm license and correct mujoco-python bindings.
## Contributing
- Fork the repo and create a feature branch.
- Add tests for new features and update documentation.
- Open a pull request with a clear description and rationale.
## Contact
Project owner: Bibek Poudel
