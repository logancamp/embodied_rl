#!/bin/bash
# setup_windows.sh — Run this inside WSL2 on your Windows machine
#
# What this does:
#   1. Checks Docker + NVIDIA Container Toolkit are installed
#   2. Verifies GPU is visible inside Docker
#   3. Builds the DreamerV3 image
#   4. Runs a quick sanity check
#
# Prerequisites (install these manually before running this script):
#   - WSL2 with Ubuntu 22.04
#   - Docker Desktop for Windows (with WSL2 backend enabled)
#   - NVIDIA drivers on Windows (not inside WSL — Windows driver serves WSL)
#   - NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

set -e

echo "================================================"
echo " DreamerV3 Windows/WSL2 Setup"
echo "================================================"

# ── Check Docker ───────────────────────────────────────────────────────────────
echo ""
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found."
    echo "Install Docker Desktop for Windows, enable WSL2 backend, then re-run."
    exit 1
fi
echo "  Docker: $(docker --version)"

# ── Check NVIDIA GPU ───────────────────────────────────────────────────────────
echo ""
echo "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found in WSL."
    echo "Make sure:"
    echo "  1. NVIDIA drivers are installed on Windows (not WSL)"
    echo "  2. You are using WSL2 (not WSL1)"
    echo "  3. nvidia-smi is accessible: run 'nvidia-smi' manually"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | awk '{print "  GPU: " $0}'

# ── Check NVIDIA Container Toolkit ────────────────────────────────────────────
echo ""
echo "Checking NVIDIA Container Toolkit..."
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA runtime not found in Docker."
    echo "Install NVIDIA Container Toolkit:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    echo "Quick install for Ubuntu/WSL2:"
    echo "  distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
    echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
    echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
    exit 1
fi
echo "  NVIDIA runtime found ✓"

# ── Verify GPU visible inside Docker ──────────────────────────────────────────
echo ""
echo "Testing GPU inside Docker..."
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi \
    --query-gpu=name --format=csv,noheader \
    | awk '{print "  Docker GPU: " $0 " ✓"}'

# ── Build DreamerV3 image ──────────────────────────────────────────────────────
echo ""
echo "Building DreamerV3 Docker image (first time takes 5-10 mins)..."
docker compose build

# ── Quick sanity check inside container ───────────────────────────────────────
echo ""
echo "Running sanity check inside container..."
docker compose run --rm shell python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import gymnasium as gym, ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', obs_type='rgb', frameskip=1)
obs, _ = env.reset()
print(f'  Pong:     {obs.shape} ✓')
env.close()
print()
print('  All checks passed. Ready to train.')
"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo " Setup complete!"
echo ""
echo " To run training:"
echo "   docker compose up train"
echo ""
echo " To run debug smoke test:"
echo "   docker compose run debug"
echo ""
echo " To open interactive shell:"
echo "   docker compose run shell"
echo ""
echo " Logs will appear in ./logs/"
echo "================================================"