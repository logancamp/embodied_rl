# Makefile — shortcuts for common commands
# Usage: make <target>

PYTHON = python
CONFIGS_DEBUG = --configs configs/base.yaml configs/debug.yaml
CONFIGS_FULL  = --configs configs/base.yaml configs/s_size.yaml

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	chmod +x setup_mac.sh && ./setup_mac.sh

# ── Training ──────────────────────────────────────────────────────────────────

# Debug run on Mac (XS model, ~50k steps, smoke test only)
run:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) train.py $(CONFIGS_DEBUG)

# Full training run on RTX 3070 (S model, 500k steps, paper benchmark)
train:
	$(PYTHON) train.py $(CONFIGS_FULL)

# Resume from latest checkpoint
resume:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) train.py $(CONFIGS_DEBUG) --resume

resume-full:
	$(PYTHON) train.py $(CONFIGS_FULL) --resume

# ── Verification ──────────────────────────────────────────────────────────────

# Verify environment and dependencies are correctly installed
check:
	$(PYTHON) -c "\
import torch; \
print(f'PyTorch:  {torch.__version__}'); \
print(f'MPS:      {torch.backends.mps.is_available()}'); \
import gymnasium; print(f'Gymnasium: {gymnasium.__version__}'); \
import ale_py; print(f'ALE:      {ale_py.__version__}'); \
"

# Verify Atari ROMs are registered
check-roms:
	$(PYTHON) -c "\
import gymnasium as gym, ale_py; \
gym.register_envs(ale_py); \
env = gym.make('ALE/Pong-v5', obs_type='rgb', frameskip=1); \
obs, _ = env.reset(); \
print(f'Pong OK — obs shape: {obs.shape}'); \
env.close() \
"

# Quick syntax check all Python files
lint:
	$(PYTHON) -m py_compile train.py utils/misc.py utils/logging.py \
		envs/atari.py models/encoder.py models/rssm.py models/heads.py \
		models/actor.py models/critic.py training/replay_buffer.py \
		training/world_model.py training/actor_critic.py
	@echo "All files OK"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean-logs:
	rm -rf logs/

clean-env:
	rm -rf dreamer_env/

# ── Docker (RTX 3070 / Windows) ───────────────────────────────────────────────

# Build Docker image
docker-build:
	docker compose build

# Full training run inside Docker
docker-train:
	docker compose up train

# Debug run inside Docker
docker-debug:
	docker compose run debug

# Interactive shell inside container
docker-shell:
	docker compose run shell

# Check GPU is visible inside Docker
docker-check-gpu:
	docker compose run --rm shell python -c "\
import torch; \
print(f'CUDA: {torch.cuda.is_available()}'); \
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}') \
"

.PHONY: setup run train resume resume-full check check-roms lint clean-logs clean-env \
        docker-build docker-train docker-debug docker-shell docker-check-gpu