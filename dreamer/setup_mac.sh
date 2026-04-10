#!/bin/bash
# setup_mac.sh — One-time Mac development environment setup
#
# Run from the dreamer/ directory:
#   chmod +x setup_mac.sh
#   ./setup_mac.sh

set -e  # exit on any error

echo "================================================"
echo " DreamerV3 Mac Setup"
echo "================================================"

# ── Python version check ──────────────────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED="3.10"
echo "Python: $PYTHON_VERSION"
if [[ "$(echo -e "$PYTHON_VERSION\n$REQUIRED" | sort -V | head -1)" != "$REQUIRED" ]]; then
    echo "ERROR: Python 3.10+ required. Install via: brew install python@3.11"
    exit 1
fi

# ── Create virtual environment ────────────────────────────────────────────────
if [ ! -d "dreamer_env" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv dreamer_env
else
    echo "Virtual environment already exists, skipping creation."
fi

# ── Activate ──────────────────────────────────────────────────────────────────
source dreamer_env/bin/activate
echo "Activated: $(which python)"

# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

# ── Register Atari ROMs ───────────────────────────────────────────────────────
echo ""
echo "Registering Atari ROMs..."
autorom --accept-license

# ── Verify MPS ────────────────────────────────────────────────────────────────
echo ""
echo "Checking MPS (Apple Silicon GPU)..."
python3 -c "
import torch
built     = torch.backends.mps.is_built()
available = torch.backends.mps.is_available()
print(f'  MPS built:     {built}')
print(f'  MPS available: {available}')
if available:
    x = torch.ones(3, device='mps')
    print(f'  MPS tensor:    {x}  ✓')
    print('  GPU acceleration ENABLED')
else:
    print('  WARNING: MPS not available — training will run on CPU (slow)')
    print('  Make sure you have PyTorch 2.0+ and macOS 12.3+')
"

# ── Verify Atari ──────────────────────────────────────────────────────────────
echo ""
echo "Checking Atari environment..."
python3 -c "
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', obs_type='rgb', frameskip=1)
obs, _ = env.reset()
print(f'  Pong obs shape: {obs.shape}  ✓')
env.close()
print('  Atari READY')
"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo " Setup complete!"
echo ""
echo " To activate this environment in future sessions:"
echo "   source dreamer_env/bin/activate"
echo ""
echo " To run training:"
echo "   PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py \\"
echo "     --configs configs/base.yaml configs/debug.yaml"
echo "================================================"