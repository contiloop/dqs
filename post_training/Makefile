SHELL := /bin/sh

PYTHON ?= python3
LAUNCH ?= $(PYTHON)
MODEL_DIR ?= models/sft_final
MPO_CONFIG ?= configs/mpo.yaml
CPO_CONFIG ?= configs/cpo.yaml
DPO_CONFIG ?= configs/dpo.yaml
TRANSFORMERS_OVERRIDE_REQUIREMENTS ?= requirements-transformers-gemma4.txt
DOWNLOAD_WORKERS ?= 8
DOWNLOAD_REPLACE ?= 0
MODEL_DOWNLOAD_WORKERS ?= 8
MODEL_REPLACE ?= 0
SET_MODEL := --set model.name_or_path=$(MODEL_DIR)
UNSLOTH_RUNTIME_ENV := UNSLOTH_COMPILE_DISABLE=1

.PHONY: set validate download-data download-model download-all validate-data validate-runtime validate-model test test-all \
	test-runtime test-research test-release build-release \
	dry-run dry-run-mpo dry-run-cpo dry-run-dpo \
	smoke-mpo smoke-cpo smoke-dpo \
	train-mpo train-cpo train-dpo

HF_DATASET_REPO := alwaysgood/dqs-post-training
HF_DATASET_REVISION := 0f7b051f96b3ccdc3837f9537e5aac3a776bf4f1
RELEASE_DIR ?= dist/dqs_preference_training
TEST_PYTHONPATH := $(abspath src):$(abspath research/data_prep):$(abspath research/audits):$(abspath research/evaluation)

set:
	$(PYTHON) -m pip install -r requirements-gpu.txt
	$(PYTHON) -m pip install --no-deps --force-reinstall -r $(TRANSFORMERS_OVERRIDE_REQUIREMENTS)

validate:
	$(PYTHON) scripts/validate_bundle.py

download-data: validate
	$(PYTHON) scripts/download_data.py \
		--workers $(DOWNLOAD_WORKERS) $(if $(filter 1,$(DOWNLOAD_REPLACE)),--replace,)

download-model: validate
	$(PYTHON) scripts/download_model.py \
		--output $(MODEL_DIR) \
		--workers $(MODEL_DOWNLOAD_WORKERS) $(if $(filter 1,$(MODEL_REPLACE)),--replace,)

download-all: download-data download-model

validate-data:
	$(PYTHON) scripts/download_data.py --check

validate-runtime:
	$(PYTHON) scripts/validate_bundle.py --runtime

validate-model:
	$(PYTHON) scripts/validate_bundle.py --model-dir $(MODEL_DIR)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

test-all: test test-runtime test-research test-release

test-runtime:
	PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m unittest discover -s tests/runtime -p 'test_*.py' -v

test-research:
	PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m unittest discover -s research/tests -p 'test_*.py' -v

test-release:
	PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m unittest discover -s tests/release -p 'test_*.py' -v

build-release:
	$(PYTHON) scripts/build_release.py \
		--output $(RELEASE_DIR) \
		--data-mode hf \
		--hf-repo-id $(HF_DATASET_REPO) \
		--hf-revision $(HF_DATASET_REVISION) \
		--replace

dry-run: dry-run-mpo dry-run-cpo dry-run-dpo

dry-run-mpo: validate-data
	PYTHONPATH=src $(PYTHON) src/train_mpo.py --config $(MPO_CONFIG) --dry-run $(SET_MODEL)

dry-run-cpo: validate-data
	PYTHONPATH=src $(PYTHON) src/train_cpo.py --config $(CPO_CONFIG) --dry-run $(SET_MODEL)

dry-run-dpo: validate-data
	PYTHONPATH=src $(PYTHON) src/train_dpo.py --config $(DPO_CONFIG) --dry-run $(SET_MODEL)

smoke-mpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(UNSLOTH_RUNTIME_ENV) $(LAUNCH) src/train_mpo.py --config $(MPO_CONFIG) --smoke-step $(SET_MODEL)

smoke-cpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(UNSLOTH_RUNTIME_ENV) $(LAUNCH) src/train_cpo.py --config $(CPO_CONFIG) --smoke-step $(SET_MODEL)

smoke-dpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(UNSLOTH_RUNTIME_ENV) $(LAUNCH) src/train_dpo.py --config $(DPO_CONFIG) --smoke-step $(SET_MODEL)

train-mpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(UNSLOTH_RUNTIME_ENV) $(LAUNCH) src/train_mpo.py --config $(MPO_CONFIG) $(SET_MODEL)

train-cpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(UNSLOTH_RUNTIME_ENV) $(LAUNCH) src/train_cpo.py --config $(CPO_CONFIG) $(SET_MODEL)

train-dpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(UNSLOTH_RUNTIME_ENV) $(LAUNCH) src/train_dpo.py --config $(DPO_CONFIG) $(SET_MODEL)
