SHELL := /bin/sh
PYTHON_VERSION ?= 3.11
PYTHON ?= python$(PYTHON_VERSION)
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PYTHON_TAG := cp$(subst .,,$(PYTHON_VERSION))
USE_VENV ?= 0
PY := $(if $(filter 1,$(USE_VENV)),$(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON)),$(PYTHON))
REAL_ENV_PY := $(if $(filter 1,$(USE_VENV)),$(VENV_PYTHON),$(PYTHON))
SETUP_PY := $(if $(filter 1,$(USE_VENV)),$(VENV_PYTHON),$(PYTHON))
QE_VENV_DIR ?= $(HOME)/.venvs/comet
COMET_PYTHON ?= $(QE_VENV_DIR)/bin/python
METRICX_VENV_DIR ?= $(HOME)/.venvs/metricx
METRICX_PYTHON ?= $(METRICX_VENV_DIR)/bin/python
METRICX_REPO_DIR ?= $(abspath ../metricx)
DQS_QUIET ?= 1
DQS_PROGRESS ?= 1
DQS_SHOW_UNSLOTH_LOGS ?= 1
SKIP_CAUSAL_CONV1D ?= 0
SKIP_METRICX ?= 0
DATA_CONFIG ?= configs/data.yaml
HF_DATASET_REPO ?=
HF_DATASET_REVISION ?=
HF_DATASET_LOCAL_DIR ?=
HF_DOWNLOAD_WORKERS ?=
HF_DOWNLOAD_DRY_RUN ?= 0
PREPROCESS_WORKERS ?=
PREPROCESS_TOKENIZER_MODEL ?=
PREPROCESS_OUTPUT_DIR ?=
PREPROCESS_LIMIT ?=
TRAIN_CONFIG ?= configs/config.yaml
TRAIN_SUBSET_IDX ?=
TRAIN_SUBSET_SIZE ?=
TRAIN_DATA_PATH ?=
TRAIN_START_FROM ?=
TRAIN_RESUME ?= auto
TRAIN_DRY_RUN ?= 0
TRAIN_FORCE ?= 0
TRAIN_OVERRIDES ?=
TRAIN_STAGE_END_SUBSET ?=
TRAIN_STAGE_MAX_SUBSETS ?=
EVAL_CONFIG ?= configs/config.yaml
EVAL_PROFILE ?= val
EVAL_DATA_PATH ?=
EVAL_MODEL_PATH ?=
EVAL_OUTPUT_DIR ?=
EVAL_LIMIT ?=
EVAL_METRICS ?=
EVAL_DRY_RUN ?= 0
EVAL_FORCE ?= 0
EVAL_EVERY_N_SUBSETS ?= 1
EVAL_ON_FINAL_SUBSET ?= 1
EVAL_OVERRIDES ?=
EVAL_MATRIX_CONFIG ?= configs/eval_matrix/openrouter.yaml
EVAL_MATRIX_OUTPUT_DIR ?=
EVAL_MATRIX_PROFILE ?=
EVAL_MATRIX_MODELS ?=
EVAL_MATRIX_DATA_PATH ?=
EVAL_MATRIX_LIMIT ?=
EVAL_MATRIX_METRICS ?=
EVAL_MATRIX_MAX_NEW_TOKENS ?=
EVAL_MATRIX_DRY_RUN ?= 0
EVAL_MATRIX_FORCE ?= 0
EVAL_CHECKPOINT_PROFILE ?= final
EVAL_CHECKPOINT_DIR ?=
EVAL_CHECKPOINT_OUTPUT_DIR ?=
EVAL_CHECKPOINT_START_STEP ?=
EVAL_CHECKPOINT_END_STEP ?=
EVAL_CHECKPOINT_MAX ?=
UPLOAD_CONFIG ?= configs/config.yaml
UPLOAD_RUN_ID ?=
UPLOAD_RUN_DIR ?=
UPLOAD_PATH_IN_REPO ?=
UPLOAD_REVISION ?= main
UPLOAD_PRIVATE ?= 1
UPLOAD_CREATE_REPO ?= 1
UPLOAD_DRY_RUN ?= 0
UPLOAD_DELETE_EXISTING ?= 0
UPLOAD_COMMIT_MESSAGE ?=
UPLOAD_OVERRIDES ?=
UPLOAD_IGNORE_PATTERNS ?=
COMPACT_CONFIG ?= configs/config.yaml
COMPACT_RUN_ID ?=
COMPACT_RUN_DIR ?=
COMPACT_DRY_RUN ?= 0
COMPACT_OVERRIDES ?=
FULL_SFT_CONFIG ?= configs/config.yaml
FULL_SFT_SOURCE_RUN_DIR ?=
FULL_SFT_RUN_ID ?=
FULL_SFT_START_SUBSET ?=
FULL_SFT_END_SUBSET ?=
FULL_SFT_MAX_SUBSETS ?=
FULL_SFT_OVERRIDES ?=
FULL_SFT_COPY_DATASETS ?= 1
FULL_SFT_FINAL_ONLY ?= 1
FULL_SFT_DELETE_CHECKPOINTS_ON_COMPLETE ?= 0
FULL_SFT_SINGLE_PASS ?= 1
FULL_SFT_PRESERVE_STAGE_BOUNDARIES ?= 1
FULL_SFT_DRY_RUN ?= 0
FULL_SFT_FORCE ?= 0
FULL_SFT_PLAN_ONLY ?= 0
REPAIR_MODEL_DIR ?=
REPAIR_BACKUP ?= 0
REPAIR_DRY_RUN ?= 0
SFT_CONFIG ?= configs/config.yaml
SFT_SUBSET_IDX ?=
SFT_DATASET_PATH ?=
SFT_OUTPUT_DIR ?=
SFT_DRY_RUN ?= 0
SFT_NPROC_PER_NODE ?= 1
SFT_OVERRIDES ?=
SMOKE_CONFIG ?= configs/config.yaml
SMOKE_OUTPUT_DIR ?= data/smoke
SMOKE_OVERRIDES ?=
SMOKE_LOCAL_FILES_ONLY ?= 0
SMOKE_MAX_CONTEXT_ROWS ?= 4
SMOKE_CYCLE_SUBSET_SIZE ?= 4
SMOKE_CYCLE_SUBSETS ?= 2
SMOKE_VAL_ROWS ?= 4
SMOKE_SFT_OUTPUT_DIR ?= artifacts/smoke/max_context_sft
SMOKE_SFT_DRY_RUN ?= 0
SMOKE_SFT_OVERRIDES ?= run.id=smoke_max_context logging.wandb.enabled=false training.gradient_accumulation_steps=1 training.max_steps=2 training.save_strategy=no training.save_final_model=false training.save_full_model=false training.save_merged_model=false training.merge_smoke_test_required=false
SMOKE_CYCLE_DRY_RUN ?= 0
SMOKE_CYCLE_EVAL_DRY_RUN ?= 0
SMOKE_CYCLE_OVERRIDES ?= run.id=smoke_cycle logging.wandb.enabled=false data.teacher_target_per_subset=2 teacher.candidate_multiplier=2 training.gradient_accumulation_steps=1 training.save_strategy=no training.save_merged_model=false training.merge_smoke_test_required=false
SMOKE_CYCLE_EVAL_OVERRIDES ?=
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu128
PIN_TORCH_VERSION ?= 2.10.0
PIN_TORCHVISION_VERSION ?= 0.25.0
PIN_TORCHAUDIO_VERSION ?= 2.10.0
PIN_TRANSFORMERS_VERSION ?= 5.5.0
PIN_TRL_VERSION ?= 0.24.0
PIN_DATASETS_VERSION ?= 3.4.1
PIN_UNSLOTH_VERSION ?= 2026.7.2
PIN_UNSLOTH_ZOO_VERSION ?= 2026.7.2
PIN_VLLM_VERSION ?= 0.19.1
PIN_HF_HUB_VERSION ?= 1.14.0
PIN_HF_XET_VERSION ?= 1.5.0
PIN_FLASH_ATTN_VERSION ?= 2.8.3
PIN_SETUPTOOLS_SPEC ?= "setuptools>=77.0.3,<81.0.0"
# Keep numpy below 2.3 for numba compatibility while satisfying mistral-common.
PIN_NUMPY_VERSION ?= 2.2.6
# CUTLASS DSL requires protobuf 6.x; prevent dependency upgrades from pulling 7.x.
PIN_PROTOBUF_SPEC ?= "protobuf>=6.30.2,<7"
# Keep FLA aligned with torch 2.10 runtime and avoid transitive resolver drift.
PIN_FLA_CORE_VERSION ?= 0.4.2
PIN_FLASH_LINEAR_ATTENTION_VERSION ?= 0.4.2
# MetricX pins old transformers/datasets code paths.
PIN_METRICX_PYARROW_VERSION ?= 20.0.0
PIN_METRICX_PROTOBUF_VERSION ?= 3.20.3
PIN_METRICX_FSSPEC_VERSION ?= 2023.6.0
PIN_METRICX_NUMPY_VERSION ?= 1.26.4

export DQS_QUIET
export DQS_PROGRESS
export DQS_SHOW_UNSLOTH_LOGS
export VLLM_LOGGING_LEVEL ?= ERROR
export VLLM_LOGGING_COLOR ?= 0
export TRANSFORMERS_VERBOSITY ?= error
export TOKENIZERS_PARALLELISM ?= false
export PYTHONWARNINGS ?= ignore
export WANDB_SILENT ?= true

# FlashAttention2 wheel hosted in a HF dataset.
# - FLASH_ATTN_GPU_ARCH: auto | sm80 | sm120 | default
# - FLASH_ATTN_WHL_SM80 / FLASH_ATTN_WHL_SM120: arch-specific wheel names
# - FLASH_ATTN_WHL: fallback/default wheel name
FLASH_ATTN_REPO ?= alwaysgood/scp-stage4-wheels
FLASH_ATTN_WHL ?= flash_attn-$(PIN_FLASH_ATTN_VERSION)-$(PYTHON_TAG)-$(PYTHON_TAG)-linux_x86_64.whl
FLASH_ATTN_WHL_SM80 ?= flash_attn-$(PIN_FLASH_ATTN_VERSION)-1sm80-$(PYTHON_TAG)-$(PYTHON_TAG)-linux_x86_64.whl
FLASH_ATTN_WHL_SM120 ?= flash_attn-$(PIN_FLASH_ATTN_VERSION)-1sm120-$(PYTHON_TAG)-$(PYTHON_TAG)-linux_x86_64.whl
FLASH_ATTN_GPU_ARCH ?= auto

.PHONY: set set-metricx validate-setup download-prepared-data preprocess-raw train train-stage eval eval-matrix eval-checkpoints upload-run compact-run full-sft-from-run repair-qwen35-checkpoint sft smoke-data smoke-sft-max-context smoke-cycle verify-cuda-kernels

# Target: set
# required config keys: none
# input artifacts: none
# output artifacts: local directories and selected python environments
# runtime: local/remote machine setup step; downloads packages and may require CUDA-compatible wheels
# exit behavior: 0 on successful dependency install; non-zero on package resolver/install failure
set:
	@mkdir -p artifacts/runs tests/fixtures src
	@if [ "$(USE_VENV)" = "1" ] && [ ! -x "$(VENV_PYTHON)" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv --python $(PYTHON_VERSION) --seed $(VENV_DIR); \
		else \
			$(PYTHON) -m venv $(VENV_DIR); \
		fi; \
	fi
	@if ! $(SETUP_PY) -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pytest") else 1)'; then \
		$(SETUP_PY) -m pip install --upgrade pip pytest; \
	fi
	@$(SETUP_PY) -c 'import sys; print("set:", sys.executable, sys.version.split()[0])'
	@if [ "$(USE_VENV)" = "1" ] && [ ! -x "$(VENV_PYTHON)" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv --python $(PYTHON_VERSION) --seed $(VENV_DIR); \
		else \
			$(PYTHON) -m venv $(VENV_DIR); \
		fi; \
	fi
	@$(REAL_ENV_PY) -c 'import sys; want=tuple(map(int, "$(PYTHON_VERSION)".split(".")[:2])); print("set-real-env: python", sys.version.split()[0]); sys.exit(f"set-real-env requires Python {want[0]}.{want[1]}, got {sys.version.split()[0]}") if sys.version_info[:2] != want else sys.exit(0)'
	@$(REAL_ENV_PY) -m pip install --upgrade pip
	@$(REAL_ENV_PY) -m pip install $(PIN_SETUPTOOLS_SPEC)
	@$(REAL_ENV_PY) -m pip install \
		--index-url $(TORCH_INDEX_URL) \
		"torch==$(PIN_TORCH_VERSION)" \
		"torchvision==$(PIN_TORCHVISION_VERSION)" \
		"torchaudio==$(PIN_TORCHAUDIO_VERSION)"
	@$(REAL_ENV_PY) -m pip install \
		"trl==$(PIN_TRL_VERSION)" \
		"datasets==$(PIN_DATASETS_VERSION)"
	@$(REAL_ENV_PY) -m pip install \
		"unsloth-zoo==$(PIN_UNSLOTH_ZOO_VERSION)" \
		"unsloth==$(PIN_UNSLOTH_VERSION)"
	@$(REAL_ENV_PY) -m pip uninstall -y vllm || true
	@$(REAL_ENV_PY) -m pip install \
		"vllm==$(PIN_VLLM_VERSION)" \
		--extra-index-url $(TORCH_INDEX_URL)
	@$(REAL_ENV_PY) -m pip install --index-url $(TORCH_INDEX_URL) "xformers==0.0.34"
	@$(REAL_ENV_PY) -m pip install \
		tokenizers hydra-core omegaconf \
		openai anthropic pydantic requests peft wandb sacrebleu \
		sentencepiece bitsandbytes hf_transfer msgspec tyro torchao ninja
	# Intentionally pin transformers 5.5.0 for Qwen3.5 architecture support
	# parity with the previous scp_stage4_sft runtime stack.
	@$(REAL_ENV_PY) -m pip install --no-deps \
		"transformers==$(PIN_TRANSFORMERS_VERSION)" \
		"huggingface_hub>=$(PIN_HF_HUB_VERSION),<2" \
		"hf-xet>=$(PIN_HF_XET_VERSION),<2"
	@$(REAL_ENV_PY) -m pip install --upgrade \
		"numpy==$(PIN_NUMPY_VERSION)" \
		$(PIN_PROTOBUF_SPEC)
	# FlashAttention2: choose wheel by GPU arch (sm80/sm120) when possible.
	@arch_choice="$(FLASH_ATTN_GPU_ARCH)"; \
	selected_whl="$(FLASH_ATTN_WHL)"; \
	py_tag="$(PYTHON_TAG)"; \
	py_ver="$$( $(REAL_ENV_PY) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' )"; \
	if [ "$$arch_choice" = "auto" ]; then \
		detected="$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk -F'.' 'BEGIN{max=0} {gsub(/[^0-9.]/,""); if ($$1 ~ /^[0-9]+$$/) {minor=$$2; if (minor !~ /^[0-9]+$$/) minor=0; val=($$1*10)+minor; if (val>max) max=val;}} END{if (max>0) printf "sm%d", max;}')"; \
		if [ -n "$$detected" ]; then arch_choice="$$detected"; else arch_choice="default"; fi; \
	fi; \
	case "$$arch_choice" in \
		sm80) selected_whl="$(FLASH_ATTN_WHL_SM80)" ;; \
		sm120) selected_whl="$(FLASH_ATTN_WHL_SM120)" ;; \
		default|'') selected_whl="$(FLASH_ATTN_WHL)" ;; \
		*) echo "  [WARN] unknown FLASH_ATTN_GPU_ARCH=$$arch_choice, using default wheel"; selected_whl="$(FLASH_ATTN_WHL)" ;; \
	esac; \
	echo "  flash_attn target: python=$$py_ver ($$py_tag) arch=$$arch_choice wheel=$$selected_whl"; \
	if $(REAL_ENV_PY) -m pip install \
		"https://huggingface.co/datasets/$(FLASH_ATTN_REPO)/resolve/main/$$selected_whl"; then \
		echo "  flash_attn wheel install ok: $$selected_whl"; \
	else \
		echo "  [ERROR] flash_attn wheel unavailable: $$selected_whl"; \
		exit 1; \
	fi
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip causal_conv1d setup (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(REAL_ENV_PY) bash scripts/ensure_causal_conv1d.sh; \
	fi
	@$(REAL_ENV_PY) -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule" 2>/dev/null \
		|| $(REAL_ENV_PY) -m pip install --no-deps \
			"fla-core==$(PIN_FLA_CORE_VERSION)" \
			"flash-linear-attention==$(PIN_FLASH_LINEAR_ATTENTION_VERSION)"
	@$(REAL_ENV_PY) -m pip install --upgrade \
		"numpy==$(PIN_NUMPY_VERSION)" \
		$(PIN_PROTOBUF_SPEC)
	@$(MAKE) verify-cuda-kernels REAL_ENV_PY=$(REAL_ENV_PY) SKIP_CAUSAL_CONV1D=$(SKIP_CAUSAL_CONV1D)
	@$(REAL_ENV_PY) -c 'import sys, torch; print("set-real-env:", sys.executable, "torch", torch.__version__)'
	@echo "set-real-env: setting up QE isolation venv at $(QE_VENV_DIR)..."
	@if [ ! -x "$(QE_VENV_DIR)/bin/python" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv --python $(PYTHON_VERSION) --seed $(QE_VENV_DIR); \
		else \
			$(PYTHON) -m venv --without-pip $(QE_VENV_DIR) && \
			curl -sS https://bootstrap.pypa.io/get-pip.py | $(QE_VENV_DIR)/bin/python; \
		fi; \
	fi
	@$(QE_VENV_DIR)/bin/python -m pip install --upgrade pip setuptools wheel
	@$(QE_VENV_DIR)/bin/pip install \
		torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	@$(QE_VENV_DIR)/bin/pip install --no-deps transformers
	@$(QE_VENV_DIR)/bin/pip install \
		sentencepiece safetensors accelerate huggingface_hub \
		"unbabel-comet>=2.2.7" sacrebleu
	@$(QE_VENV_DIR)/bin/python -c 'import torch; print("set-real-env: QE venv torch", torch.__version__, "cuda", torch.cuda.is_available())'
	@echo "set-real-env: COMET_PYTHON=$(COMET_PYTHON)"
	@$(MAKE) set-metricx \
		PYTHON="$(PYTHON)" \
		PYTHON_VERSION="$(PYTHON_VERSION)" \
		METRICX_VENV_DIR="$(METRICX_VENV_DIR)" \
		METRICX_REPO_DIR="$(METRICX_REPO_DIR)" \
		PIN_METRICX_PYARROW_VERSION="$(PIN_METRICX_PYARROW_VERSION)" \
		PIN_METRICX_PROTOBUF_VERSION="$(PIN_METRICX_PROTOBUF_VERSION)" \
		PIN_METRICX_FSSPEC_VERSION="$(PIN_METRICX_FSSPEC_VERSION)" \
		PIN_METRICX_NUMPY_VERSION="$(PIN_METRICX_NUMPY_VERSION)" \
		SKIP_METRICX="$(SKIP_METRICX)"

# Target: set-metricx
# required config keys: none
# input artifacts: google-research/metricx repo downloaded to METRICX_REPO_DIR
# output artifacts: MetricX isolation venv at METRICX_VENV_DIR
# runtime: local/remote machine setup step for heavyweight final eval metrics
# exit behavior: 0 on successful MetricX setup; non-zero on clone/install/import failure
set-metricx:
ifeq ($(SKIP_METRICX),1)
	@echo "set-metricx: skip MetricX setup (SKIP_METRICX=1)"
else
	@echo "set-metricx: setting up MetricX repo at $(METRICX_REPO_DIR)..."
	@if [ -d "$(METRICX_REPO_DIR)/.git" ]; then \
		echo "set-metricx: repo exists"; \
	elif [ -e "$(METRICX_REPO_DIR)" ]; then \
		echo "set-metricx: $(METRICX_REPO_DIR) exists but is not a git checkout"; \
		exit 1; \
	else \
		git clone https://github.com/google-research/metricx "$(METRICX_REPO_DIR)"; \
	fi
	@echo "set-metricx: setting up MetricX isolation venv at $(METRICX_VENV_DIR)..."
	@if [ ! -x "$(METRICX_VENV_DIR)/bin/python" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv --python $(PYTHON_VERSION) --seed "$(METRICX_VENV_DIR)"; \
		else \
			$(PYTHON) -m venv "$(METRICX_VENV_DIR)"; \
		fi; \
	fi
	@$(METRICX_VENV_DIR)/bin/python -m pip install --upgrade pip setuptools wheel
	@constraints="$$(mktemp)"; \
	echo "pyarrow==$(PIN_METRICX_PYARROW_VERSION)" > "$$constraints"; \
	echo "protobuf==$(PIN_METRICX_PROTOBUF_VERSION)" >> "$$constraints"; \
	echo "fsspec==$(PIN_METRICX_FSSPEC_VERSION)" >> "$$constraints"; \
	echo "numpy==$(PIN_METRICX_NUMPY_VERSION)" >> "$$constraints"; \
	$(METRICX_VENV_DIR)/bin/python -m pip install -r "$(METRICX_REPO_DIR)/requirements.txt" -c "$$constraints"; \
	status="$$?"; \
	rm -f "$$constraints"; \
	exit "$$status"
	@PYTHONPATH="$(METRICX_REPO_DIR)" $(METRICX_VENV_DIR)/bin/python -c 'import numpy as np; import pyarrow as pa; assert hasattr(pa, "PyExtensionType"), pa.__version__; import google.protobuf as pb; import fsspec; import metricx24.predict; print("set-metricx: numpy", np.__version__); print("set-metricx: pyarrow", pa.__version__); print("set-metricx: protobuf", pb.__version__); print("set-metricx: fsspec", fsspec.__version__); print("set-metricx: METRICX_PYTHON=$(METRICX_VENV_DIR)/bin/python"); print("set-metricx: METRICX_REPO_DIR=$(METRICX_REPO_DIR)")'
endif

# Target: verify-cuda-kernels
# required config keys: none
# input artifacts: installed runtime python packages in REAL_ENV_PY
# output artifacts: stdout kernel readiness status
# runtime: local/remote GPU runtime check
# exit behavior: 0 if kernel checks pass (or skip flag enabled); non-zero on kernel check failure
verify-cuda-kernels:
	@if [ "$(SKIP_CAUSAL_CONV1D)" = "1" ]; then \
		echo "  skip CUDA kernel verification (SKIP_CAUSAL_CONV1D=1)"; \
	else \
		PYTHON=$(REAL_ENV_PY) bash scripts/verify_cuda_kernels.sh; \
	fi

validate-setup:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	"$$py" scripts/validate_setup.py

download-prepared-data:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	"$$py" scripts/download_prepared_data.py \
		--config "$(DATA_CONFIG)" \
		$(if $(HF_DATASET_REPO),--repo "$(HF_DATASET_REPO)",) \
		$(if $(HF_DATASET_REVISION),--revision "$(HF_DATASET_REVISION)",) \
		$(if $(HF_DATASET_LOCAL_DIR),--local-dir "$(HF_DATASET_LOCAL_DIR)",) \
		$(if $(HF_DOWNLOAD_WORKERS),--workers "$(HF_DOWNLOAD_WORKERS)",) \
		$(if $(filter 1,$(HF_DOWNLOAD_DRY_RUN)),--dry-run,)

preprocess-raw:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src "$$py" src/preprocess_raw.py \
		--config "$(DATA_CONFIG)" \
		$(if $(PREPROCESS_WORKERS),--workers "$(PREPROCESS_WORKERS)",) \
		$(if $(PREPROCESS_TOKENIZER_MODEL),--tokenizer-model "$(PREPROCESS_TOKENIZER_MODEL)",) \
		$(if $(PREPROCESS_OUTPUT_DIR),--output-dir "$(PREPROCESS_OUTPUT_DIR)",) \
		$(if $(PREPROCESS_LIMIT),--limit "$(PREPROCESS_LIMIT)",)

smoke-data:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src "$$py" src/smoke_data.py \
		--config "$(SMOKE_CONFIG)" \
		--output-dir "$(SMOKE_OUTPUT_DIR)" \
		--max-context-rows "$(SMOKE_MAX_CONTEXT_ROWS)" \
		--cycle-subset-size "$(SMOKE_CYCLE_SUBSET_SIZE)" \
		--cycle-subsets "$(SMOKE_CYCLE_SUBSETS)" \
		--val-rows "$(SMOKE_VAL_ROWS)" \
		$(foreach override,$(SMOKE_OVERRIDES),--override "$(override)") \
		$(if $(filter 1,$(SMOKE_LOCAL_FILES_ONLY)),--local-files-only,)

smoke-sft-max-context: smoke-data
	@$(MAKE) sft \
		SFT_CONFIG="$(SMOKE_CONFIG)" \
		SFT_SUBSET_IDX=0 \
		SFT_DATASET_PATH="$(SMOKE_OUTPUT_DIR)/max_context_sft.jsonl" \
		SFT_OUTPUT_DIR="$(SMOKE_SFT_OUTPUT_DIR)" \
		SFT_DRY_RUN="$(SMOKE_SFT_DRY_RUN)" \
		SFT_NPROC_PER_NODE="$(SFT_NPROC_PER_NODE)" \
		SFT_OVERRIDES='$(SMOKE_OVERRIDES) $(SMOKE_SFT_OVERRIDES)'

smoke-cycle: smoke-data
	@$(MAKE) train-stage \
		TRAIN_CONFIG="$(SMOKE_CONFIG)" \
		TRAIN_DATA_PATH="$(SMOKE_OUTPUT_DIR)/cycle.jsonl" \
		TRAIN_SUBSET_SIZE="$(SMOKE_CYCLE_SUBSET_SIZE)" \
		TRAIN_STAGE_MAX_SUBSETS="$(SMOKE_CYCLE_SUBSETS)" \
		TRAIN_RESUME=none \
		TRAIN_FORCE=1 \
		TRAIN_DRY_RUN="$(SMOKE_CYCLE_DRY_RUN)" \
		EVAL_PROFILE=smoke \
		EVAL_DATA_PATH="$(SMOKE_OUTPUT_DIR)/val.jsonl" \
		EVAL_LIMIT="$(SMOKE_VAL_ROWS)" \
		EVAL_FORCE=1 \
		EVAL_DRY_RUN="$(SMOKE_CYCLE_EVAL_DRY_RUN)" \
		EVAL_EVERY_N_SUBSETS=1 \
		TRAIN_OVERRIDES='$(SMOKE_OVERRIDES) $(SMOKE_CYCLE_OVERRIDES)' \
		EVAL_OVERRIDES='$(SMOKE_OVERRIDES) $(SMOKE_CYCLE_EVAL_OVERRIDES)'

train:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src "$$py" src/train.py \
		--config "$(TRAIN_CONFIG)" \
		$(if $(TRAIN_SUBSET_IDX),--subset-idx "$(TRAIN_SUBSET_IDX)",) \
		$(if $(TRAIN_SUBSET_SIZE),--subset-size "$(TRAIN_SUBSET_SIZE)",) \
		$(if $(TRAIN_DATA_PATH),--data-path "$(TRAIN_DATA_PATH)",) \
		$(if $(TRAIN_START_FROM),--start-from-phase "$(TRAIN_START_FROM)",) \
		--resume "$(TRAIN_RESUME)" \
		--sft-nproc-per-node "$(SFT_NPROC_PER_NODE)" \
		$(foreach override,$(TRAIN_OVERRIDES),--override "$(override)") \
		$(if $(filter 1,$(TRAIN_DRY_RUN)),--dry-run,) \
		$(if $(filter 1,$(TRAIN_FORCE)),--force,)

train-stage:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src COMET_PYTHON="$(COMET_PYTHON)" METRICX_PYTHON="$(METRICX_PYTHON)" METRICX_REPO_DIR="$(METRICX_REPO_DIR)" "$$py" src/train_stage.py \
		--config "$(TRAIN_CONFIG)" \
		$(if $(TRAIN_SUBSET_IDX),--subset-idx "$(TRAIN_SUBSET_IDX)",) \
		$(if $(TRAIN_SUBSET_SIZE),--subset-size "$(TRAIN_SUBSET_SIZE)",) \
		$(if $(TRAIN_DATA_PATH),--data-path "$(TRAIN_DATA_PATH)",) \
		$(if $(TRAIN_START_FROM),--start-from-phase "$(TRAIN_START_FROM)",) \
		--resume "$(TRAIN_RESUME)" \
		--sft-nproc-per-node "$(SFT_NPROC_PER_NODE)" \
		$(if $(TRAIN_STAGE_END_SUBSET),--stage-end-subset "$(TRAIN_STAGE_END_SUBSET)",) \
		$(if $(TRAIN_STAGE_MAX_SUBSETS),--stage-max-subsets "$(TRAIN_STAGE_MAX_SUBSETS)",) \
		--eval-every-n-subsets "$(EVAL_EVERY_N_SUBSETS)" \
		--eval-config "$(EVAL_CONFIG)" \
		--eval-profile "$(EVAL_PROFILE)" \
		$(if $(EVAL_DATA_PATH),--eval-data-path "$(EVAL_DATA_PATH)",) \
		$(if $(EVAL_MODEL_PATH),--eval-model-path "$(EVAL_MODEL_PATH)",) \
		$(if $(EVAL_OUTPUT_DIR),--eval-output-dir "$(EVAL_OUTPUT_DIR)",) \
		$(if $(EVAL_LIMIT),--eval-limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_METRICS),--eval-metrics "$(EVAL_METRICS)",) \
		$(foreach override,$(TRAIN_OVERRIDES),--override "$(override)") \
		$(foreach override,$(EVAL_OVERRIDES),--eval-override "$(override)") \
		$(if $(filter 1,$(TRAIN_DRY_RUN)),--dry-run,) \
		$(if $(filter 1,$(TRAIN_FORCE)),--force,) \
		$(if $(filter 1,$(EVAL_DRY_RUN)),--eval-dry-run,) \
		$(if $(filter 1,$(EVAL_FORCE)),--eval-force,) \
		$(if $(filter 1,$(EVAL_ON_FINAL_SUBSET)),--eval-on-final-subset,)

eval:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src COMET_PYTHON="$(COMET_PYTHON)" METRICX_PYTHON="$(METRICX_PYTHON)" METRICX_REPO_DIR="$(METRICX_REPO_DIR)" "$$py" src/eval.py \
		--config "$(EVAL_CONFIG)" \
		--override "eval=$(EVAL_PROFILE)" \
		$(if $(EVAL_DATA_PATH),--data-path "$(EVAL_DATA_PATH)",) \
		$(if $(EVAL_MODEL_PATH),--model-path "$(EVAL_MODEL_PATH)",) \
		$(if $(EVAL_OUTPUT_DIR),--output-dir "$(EVAL_OUTPUT_DIR)",) \
		$(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_METRICS),--metrics "$(EVAL_METRICS)",) \
		$(foreach override,$(EVAL_OVERRIDES),--override "$(override)") \
		$(if $(filter 1,$(EVAL_DRY_RUN)),--dry-run,) \
		$(if $(filter 1,$(EVAL_FORCE)),--force,)

eval-matrix:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src COMET_PYTHON="$(COMET_PYTHON)" METRICX_PYTHON="$(METRICX_PYTHON)" METRICX_REPO_DIR="$(METRICX_REPO_DIR)" "$$py" src/eval_matrix.py \
		--matrix-config "$(EVAL_MATRIX_CONFIG)" \
		--eval-config "$(EVAL_CONFIG)" \
		$(if $(EVAL_MATRIX_OUTPUT_DIR),--output-dir "$(EVAL_MATRIX_OUTPUT_DIR)",) \
		$(if $(EVAL_MATRIX_PROFILE),--profile "$(EVAL_MATRIX_PROFILE)",) \
		$(if $(EVAL_MATRIX_MODELS),--models "$(EVAL_MATRIX_MODELS)",) \
		$(if $(EVAL_MATRIX_DATA_PATH),--data-path "$(EVAL_MATRIX_DATA_PATH)",) \
		$(if $(EVAL_MATRIX_LIMIT),--limit "$(EVAL_MATRIX_LIMIT)",) \
		$(if $(EVAL_MATRIX_METRICS),--metrics "$(EVAL_MATRIX_METRICS)",) \
		$(if $(EVAL_MATRIX_MAX_NEW_TOKENS),--max-new-tokens "$(EVAL_MATRIX_MAX_NEW_TOKENS)",) \
		$(if $(filter 1,$(EVAL_MATRIX_DRY_RUN)),--dry-run,) \
		$(if $(filter 1,$(EVAL_MATRIX_FORCE)),--force,)

eval-checkpoints:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src COMET_PYTHON="$(COMET_PYTHON)" METRICX_PYTHON="$(METRICX_PYTHON)" METRICX_REPO_DIR="$(METRICX_REPO_DIR)" "$$py" src/eval_checkpoints.py \
		--config "$(EVAL_CONFIG)" \
		--profile "$(EVAL_CHECKPOINT_PROFILE)" \
		$(if $(EVAL_CHECKPOINT_DIR),--checkpoint-dir "$(EVAL_CHECKPOINT_DIR)",) \
		$(if $(EVAL_CHECKPOINT_OUTPUT_DIR),--output-dir "$(EVAL_CHECKPOINT_OUTPUT_DIR)",) \
		$(if $(EVAL_DATA_PATH),--data-path "$(EVAL_DATA_PATH)",) \
		$(if $(EVAL_LIMIT),--limit "$(EVAL_LIMIT)",) \
		$(if $(EVAL_METRICS),--metrics "$(EVAL_METRICS)",) \
		$(if $(EVAL_CHECKPOINT_START_STEP),--start-step "$(EVAL_CHECKPOINT_START_STEP)",) \
		$(if $(EVAL_CHECKPOINT_END_STEP),--end-step "$(EVAL_CHECKPOINT_END_STEP)",) \
		$(if $(EVAL_CHECKPOINT_MAX),--max-checkpoints "$(EVAL_CHECKPOINT_MAX)",) \
		$(foreach override,$(EVAL_OVERRIDES),--eval-override "$(override)") \
		$(if $(filter 1,$(EVAL_DRY_RUN)),--dry-run,) \
		$(if $(filter 1,$(EVAL_FORCE)),--force,)

upload-run:
	@if [ -z "$(HF_DATASET_REPO)" ]; then \
		echo "HF_DATASET_REPO is required, e.g. make upload-run HF_DATASET_REPO=username/dqs-runs UPLOAD_RUN_ID=qwen3.5_4b_instruct_lora_sf_on_seed42"; \
		exit 2; \
	fi
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src "$$py" src/upload_run.py \
		--config "$(UPLOAD_CONFIG)" \
		--repo "$(HF_DATASET_REPO)" \
		--revision "$(UPLOAD_REVISION)" \
		$(if $(UPLOAD_RUN_ID),--run-id "$(UPLOAD_RUN_ID)",) \
		$(if $(UPLOAD_RUN_DIR),--run-dir "$(UPLOAD_RUN_DIR)",) \
		$(if $(UPLOAD_PATH_IN_REPO),--path-in-repo "$(UPLOAD_PATH_IN_REPO)",) \
		$(if $(UPLOAD_COMMIT_MESSAGE),--commit-message "$(UPLOAD_COMMIT_MESSAGE)",) \
		$(foreach override,$(UPLOAD_OVERRIDES),--override "$(override)") \
		$(foreach pattern,$(UPLOAD_IGNORE_PATTERNS),--ignore-pattern "$(pattern)") \
		$(if $(filter 1,$(UPLOAD_CREATE_REPO)),--create-repo,) \
		$(if $(filter 1,$(UPLOAD_PRIVATE)),--private,) \
		$(if $(filter 1,$(UPLOAD_DELETE_EXISTING)),--delete-existing-path,) \
		$(if $(filter 1,$(UPLOAD_DRY_RUN)),--dry-run,)

compact-run:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src "$$py" src/compact_run.py \
		--config "$(COMPACT_CONFIG)" \
		$(if $(COMPACT_RUN_ID),--run-id "$(COMPACT_RUN_ID)",) \
		$(if $(COMPACT_RUN_DIR),--run-dir "$(COMPACT_RUN_DIR)",) \
		$(foreach override,$(COMPACT_OVERRIDES),--override "$(override)") \
		$(if $(filter 1,$(COMPACT_DRY_RUN)),--dry-run,)

full-sft-from-run:
	@if [ -z "$(FULL_SFT_SOURCE_RUN_DIR)" ]; then \
		echo "FULL_SFT_SOURCE_RUN_DIR is required, e.g. make full-sft-from-run FULL_SFT_SOURCE_RUN_DIR=artifacts/runs/qwen35_4b_it_lora_seed42 FULL_SFT_RUN_ID=qwen35_4b_it_full_seed42"; \
		exit 1; \
	fi
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src "$$py" src/full_sft_from_run.py \
		--config "$(FULL_SFT_CONFIG)" \
		--source-run-dir "$(FULL_SFT_SOURCE_RUN_DIR)" \
		$(if $(FULL_SFT_RUN_ID),--run-id "$(FULL_SFT_RUN_ID)",) \
		$(if $(FULL_SFT_START_SUBSET),--subset-idx "$(FULL_SFT_START_SUBSET)",) \
		$(if $(FULL_SFT_END_SUBSET),--stage-end-subset "$(FULL_SFT_END_SUBSET)",) \
		$(if $(FULL_SFT_MAX_SUBSETS),--stage-max-subsets "$(FULL_SFT_MAX_SUBSETS)",) \
		--sft-nproc-per-node "$(SFT_NPROC_PER_NODE)" \
		$(foreach override,$(FULL_SFT_OVERRIDES),--override "$(override)") \
		$(if $(filter 0,$(FULL_SFT_COPY_DATASETS)),--no-copy-datasets,) \
		$(if $(filter 1,$(FULL_SFT_SINGLE_PASS)),--single-pass,) \
		$(if $(filter 1,$(FULL_SFT_PRESERVE_STAGE_BOUNDARIES)),--preserve-stage-boundaries,) \
		$(if $(filter 1,$(FULL_SFT_FINAL_ONLY)),--final-only-artifacts,) \
		$(if $(filter 1,$(FULL_SFT_DELETE_CHECKPOINTS_ON_COMPLETE)),--delete-checkpoints-on-complete,) \
		$(if $(filter 1,$(FULL_SFT_DRY_RUN)),--dry-run,) \
		$(if $(filter 1,$(FULL_SFT_FORCE)),--force,) \
		$(if $(filter 1,$(FULL_SFT_PLAN_ONLY)),--plan-only,)

repair-qwen35-checkpoint:
	@if [ -z "$(REPAIR_MODEL_DIR)" ]; then \
		echo "REPAIR_MODEL_DIR is required, e.g. make repair-qwen35-checkpoint REPAIR_MODEL_DIR=artifacts/runs/<run>/checkpoints/final"; \
		exit 1; \
	fi
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	PYTHONPATH=src "$$py" src/qwen35_checkpoint_keys.py \
		--model-dir "$(REPAIR_MODEL_DIR)" \
		$(if $(filter 1,$(REPAIR_BACKUP)),--backup,) \
		$(if $(filter 1,$(REPAIR_DRY_RUN)),--dry-run,)

sft:
	@py="$(REAL_ENV_PY)"; \
	if ! command -v "$$py" >/dev/null 2>&1 && [ ! -x "$$py" ]; then \
		py=python3; \
	fi; \
	if [ "$(SFT_NPROC_PER_NODE)" != "1" ]; then \
		PYTHONPATH=src "$$py" -m torch.distributed.run --standalone --nproc_per_node "$(SFT_NPROC_PER_NODE)" src/sft_train.py \
			--config "$(SFT_CONFIG)" \
			$(if $(SFT_SUBSET_IDX),--subset-idx "$(SFT_SUBSET_IDX)",) \
			$(if $(SFT_DATASET_PATH),--dataset-path "$(SFT_DATASET_PATH)",) \
			$(if $(SFT_OUTPUT_DIR),--output-dir "$(SFT_OUTPUT_DIR)",) \
			$(foreach override,$(SFT_OVERRIDES),--override "$(override)") \
			$(if $(filter 1,$(SFT_DRY_RUN)),--dry-run,); \
	else \
		PYTHONPATH=src "$$py" src/sft_train.py \
			--config "$(SFT_CONFIG)" \
			$(if $(SFT_SUBSET_IDX),--subset-idx "$(SFT_SUBSET_IDX)",) \
			$(if $(SFT_DATASET_PATH),--dataset-path "$(SFT_DATASET_PATH)",) \
			$(if $(SFT_OUTPUT_DIR),--output-dir "$(SFT_OUTPUT_DIR)",) \
			$(foreach override,$(SFT_OVERRIDES),--override "$(override)") \
			$(if $(filter 1,$(SFT_DRY_RUN)),--dry-run,); \
	fi
