.PHONY: install run tune tune-vector tune-image tune-all board report clean test

export PYTHONUNBUFFERED=1

VENV   := ppo_env
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# ── Vector Environments ────────────────────────────────────────
V_ENVS := \
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
	InvertedPendulum-v4 \
	InvertedDoublePendulum-v4

# ── Image Environments ─────────────────────────────────────────
I_ENVS := \
	ALE/MsPacman-v5 \
	ALE/Pong-v5 \
	ALE/Breakout-v5 \
	ALE/SpaceInvaders-v5 \
	ALE/Asteroids-v5 \
	ALE/Pitfall-v5 \
	ALE/Centipede-v5 \
	ALE/DonkeyKong-v5 \
	CarRacing-v3

SAFE_ENV    := $(shell echo $(RUN_ENV) | tr '/' '_')
LATEST_CKPT := $(shell ls -t $(CHECKPOINT_DIR)/$(SAFE_ENV)_latest.pt 2>/dev/null | head -1)
RESUME      := $(if $(LATEST_CKPT),--checkpoint $(LATEST_CKPT),)


# ------- Setup Options -----------------------------------------------
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
	$(PIP) install autorom
	$(VENV)/bin/AutoROM --accept-license
	touch $(VENV)/bin/activate

# ------- Run Options -----------------------------------------------
CHECKPOINT_DIR     := checkpoints
RUN_ENV            := CartPole-v1

test:
	osascript -e 'tell app "Terminal" to do script "cd $(PWD) && $(MAKE) board"'
	sleep 2 && open -a "Safari" http://localhost:6006
	$(MAKE) run

run: $(VENV)/bin/activate
	@if [ -n "$(LATEST_CKPT)" ]; then \
		echo "  Resuming from $(LATEST_CKPT)"; \
	else \
		echo "  No checkpoint found, starting fresh"; \
	fi
	$(PYTHON) -u main.py --env $(RUN_ENV) $(RESUME)

compare: $(VENV)/bin/activate
	@echo "Comparing $(RUN_ENV) for $(COMPARE_STEPS) steps..."
	$(PYTHON) -u main.py --env $(RUN_ENV) --total-steps $(COMPARE_STEPS)
	$(PYTHON) -u report.py


# ------- Tuning Options -----------------------------------------------
TUNE_ENV           := CartPole-v1

TUNE_TRIALS_VECTOR := 50
TUNE_STEPS_VECTOR  := 50000
COMPARE_STEPS := 500000

TUNE_STEPS_IMAGE   := 200000
TUNE_TRIALS_IMAGE  := 20

tune: $(VENV)/bin/activate
	@echo "Tuning $(TUNE_ENV)..."
	PYTHONUNBUFFERED=1 $(PYTHON) -u main.py --tune --env $(TUNE_ENV) --trials $(TUNE_TRIALS_VECTOR) --total-steps $(TUNE_STEPS_VECTOR)

tune-vector: $(VENV)/bin/activate
	@for env in $(V_ENVS); do \
		echo "Tuning $$env..."; \
		PYTHONUNBUFFERED=1 $(PYTHON) -u main.py --tune --env $$env --trials $(TUNE_TRIALS_VECTOR) --total-steps $(TUNE_STEPS_VECTOR) || break; \
	done

tune-image: $(VENV)/bin/activate
	@for env in $(I_ENVS); do \
		echo "Tuning $$env..."; \
		PYTHONUNBUFFERED=1 $(PYTHON) -u main.py --tune --env $$env --trials $(TUNE_TRIALS_IMAGE) --total-steps $(TUNE_STEPS_IMAGE) || break; \
	done

tune-all: tune-vector tune-image

# ------- Tensor Board Options -----------------------------------------------
board:
	$(VENV)/bin/tensorboard --logdir runs/


# ── Reporting Options ───────────────────────────────────────────
COMBINED_METRICS := ~/embodied_rl/combined_metrics
PPO_METRICS      := ~/embodied_rl/PPO/metrics
LSTM_METRICS     := ~/embodied_rl/PPO-LSTM/metrics/ppo_lstm

report: $(VENV)/bin/activate
	$(PYTHON) -u report.py

combine: 
	mkdir -p $(COMBINED_METRICS)
	@if [ -d "$(PPO_METRICS)" ]; then cp -r $(PPO_METRICS)/. $(COMBINED_METRICS)/; fi
	@if [ -d "$(LSTM_METRICS)" ]; then cp -r $(LSTM_METRICS) $(COMBINED_METRICS)/ppo_lstm; fi
	@echo "  Combined metrics → $(COMBINED_METRICS)"

compare-all: combine
	$(PYTHON) -u report.py --metrics-dir $(COMBINED_METRICS)