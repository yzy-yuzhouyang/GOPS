# Taming the Aleatoric Impulse in Off-Policy Reinforcement Learning

This repository contains the official implementation of **DSAC-AID**. 
The code is built upon the **GOPS** framework (General Optimal control Problem Solver).
To respect the original license, we have retained the original file headers containing author names (e.g., "GOPS Team"). 
**Please note that these identities belong to the original framework developers, NOT the authors of this double-blind submission.**

## 1. Installation

### 1.1 Basic Installation (CPU-only)
Navigate to the root directory and set up the base environment using Conda. By default, this installs the CPU version of PyTorch.

```bash
cd GOPS

# Create the environment from the provided YAML file (for Linux)
conda env create -f gops_environment.nix.yml
# Create the environment from the provided YAML file (for Windows)
conda env create -f gops_environment.win.yml

# Initialize Conda (if not already initialized)
conda init
source ~/.bashrc

# Activate the environment and install the package
conda activate gops
pip install -e .
```

### 1.2 GPU Acceleration (Optional but Recommended)
To enable CUDA support, you must replace the CPU-only PyTorch with the CUDA-enabled version.
Note: The example below uses CUDA 12.4. Please adjust the URL according to your specific CUDA version.
```bash
conda activate gops

# Uninstall CPU versions
pip uninstall torch torchvision torchaudio -y

# Install CUDA versions
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```
## 2. Benchmark Support

### Option A: DeepMind Control Suite (DMC)

```bash
# 1. Clone the base environment to a new env named 'gops_dmc'
conda create --name gops_dmc --clone gops
conda activate gops_dmc

# 2. Install DMC-specific dependencies
pip install dm-control==1.0.16 mujoco==3.1.2 numpy==1.24.4
```

### Option B: OpenAI Gym MuJoCo
Prerequisites:
You must download the legacy MuJoCo binaries (e.g., mujoco210) and extract them to ~/.mujoco/mujoco210.
```bash
# 1. Clone the base environment (naming it gops_gym to avoid conflict with DMC)
conda create --name gops_mujoco --clone gops
conda activate gops_gym

# 2. Ensure gym and mujoco-py are installed
pip install gym==0.26.2 mujoco-py==2.1.2.14
```
## 3. Training
Training scripts are located in the `example_train/dsacaid/` directory.

### 3.1 Training Modes
We provide two sampling modes, distinguished by the script filename suffix:

- Parallel Sampling (_vecoffserial.py): Uses vectorized environments to collect data in parallel. (Recommended)

- Serial Sampling (_offserial.py): Uses a single environment instance for serial data collection.

### 3.2 Usage Examples
Example 1: Serial Training on DMC Humanoid
```bash
conda activate gops_dmc
python example_train/dsacaid/dsacaid_mlp_dmc_vecoffserial.py
```
Example 2: Parallel Training on Gym-MuJoCo
```bash
conda activate gops_mujoco
python example_train/dsacaid/dsacaid_mlp_mujoco_vecoffserial.py
```
