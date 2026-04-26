.PHONY: install run tune board clean test

VENV := ppo_env
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# List of all environments to choose from (for reference)
ENVS := \
	CartPole-v1 \
	Pendulum-v1 \
	Acrobot-v1 \
	MountainCar-v0 \
	MountainCarContinuous-v0 \
	LunarLander-v3 \
	LunarLanderContinuous-v3 \
	BipedalWalker-v3 \
	BipedalWalkerHardcore-v3 \
	HalfCheetah-v4 \
	Hopper-v4 \
	Walker2d-v4 \
	Ant-v4 \
	Humanoid-v4 \
	HumanoidStandup-v4 \
	Swimmer-v4 \
	Reacher-v4 \
	Pusher-v4 \
	InvertedPendulum-v4 \
	InvertedDoublePendulum-v4

# ── Configure these ────────────────────────────────────────────
RUN_ENV := LunarLander-v3 # Pendulum-v1 # CartPole-v1 # LunarLander-v3

TUNE_ENVS := Pendulum-v1 CartPole-v1 LunarLander-v3
TUNE_TRIALS := 50

CHECKPOINT_DIR := checkpoints
# ──────────────────────────────────────────────────────────────

# Auto-find the most recent checkpoint for RUN_ENV (empty if none exist)
SAFE_ENV := $(shell echo $(RUN_ENV) | tr '/' '_')
LATEST_CKPT := $(shell ls -t $(CHECKPOINT_DIR)/$(SAFE_ENV)_latest.pt 2>/dev/null | head -1)
RESUME := $(if $(LATEST_CKPT),--checkpoint $(LATEST_CKPT),)

clean:
	rm -rf $(VENV) __pycache__ runs/ ppo_env

install: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch numpy gymnasium tensorboard
	$(PIP) install "gymnasium[classic-control]"
	$(PIP) install "gymnasium[box2d]"
	$(PIP) install "gymnasium[atari]" ale-py
	$(PIP) install mujoco "gymnasium[mujoco]"
	$(PIP) install imageio opencv-python pygame
	$(PIP) install optuna
	touch $(VENV)/bin/activate

run: $(VENV)/bin/activate
	@if [ -n "$(LATEST_CKPT)" ]; then \
		echo "  Resuming from $(LATEST_CKPT)"; \
	else \
		echo "  No checkpoint found, starting fresh"; \
	fi
	$(PYTHON) -u main.py --env $(RUN_ENV) $(RESUME)

tune: $(VENV)/bin/activate
	@for env in $(TUNE_ENVS); do \
		echo "Tuning $$env..."; \
		$(PYTHON) -u main.py --tune --env $$env --trials $(TUNE_TRIALS); \
	done

board:
	$(VENV)/bin/tensorboard --logdir runs/

test:
	osascript -e 'tell app "Terminal" to do script "cd $(PWD) && $(MAKE) board"'
	sleep 2 && open -a "Safari" http://localhost:6006
	$(MAKE) run