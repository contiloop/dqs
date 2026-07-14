SHELL := /bin/sh

PYTHON ?= python3
LAUNCH ?= $(PYTHON)
MODEL_DIR ?= models/sft_final
MPO_CONFIG ?= configs/mpo.yaml
CPO_CONFIG ?= configs/cpo.yaml
DPO_CONFIG ?= configs/dpo.yaml
DOWNLOAD_WORKERS ?= 8
DOWNLOAD_REPLACE ?= 0
SET_MODEL := --set model.name_or_path=$(MODEL_DIR)

.PHONY: set validate download-data validate-data validate-runtime validate-model test \
	dry-run dry-run-mpo dry-run-cpo dry-run-dpo \
	smoke-mpo smoke-cpo smoke-dpo \
	train-mpo train-cpo train-dpo

set:
	$(PYTHON) -m pip install -r requirements-gpu.txt

validate:
	$(PYTHON) scripts/validate_bundle.py

download-data: validate
	$(PYTHON) scripts/download_data.py \
		--workers $(DOWNLOAD_WORKERS) $(if $(filter 1,$(DOWNLOAD_REPLACE)),--replace,)

validate-data:
	$(PYTHON) scripts/download_data.py --check

validate-runtime:
	$(PYTHON) scripts/validate_bundle.py --runtime

validate-model:
	$(PYTHON) scripts/validate_bundle.py --model-dir $(MODEL_DIR)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

dry-run: dry-run-mpo dry-run-cpo dry-run-dpo

dry-run-mpo: validate-data
	PYTHONPATH=src $(PYTHON) src/train_mpo.py --config $(MPO_CONFIG) --dry-run $(SET_MODEL)

dry-run-cpo: validate-data
	PYTHONPATH=src $(PYTHON) src/train_cpo.py --config $(CPO_CONFIG) --dry-run $(SET_MODEL)

dry-run-dpo: validate-data
	PYTHONPATH=src $(PYTHON) src/train_dpo.py --config $(DPO_CONFIG) --dry-run $(SET_MODEL)

smoke-mpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(PYTHON) src/train_mpo.py --config $(MPO_CONFIG) --smoke-step $(SET_MODEL)

smoke-cpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(PYTHON) src/train_cpo.py --config $(CPO_CONFIG) --smoke-step $(SET_MODEL)

smoke-dpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(PYTHON) src/train_dpo.py --config $(DPO_CONFIG) --smoke-step $(SET_MODEL)

train-mpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(LAUNCH) src/train_mpo.py --config $(MPO_CONFIG) $(SET_MODEL)

train-cpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(LAUNCH) src/train_cpo.py --config $(CPO_CONFIG) $(SET_MODEL)

train-dpo: validate-data validate-runtime validate-model
	PYTHONPATH=src $(LAUNCH) src/train_dpo.py --config $(DPO_CONFIG) $(SET_MODEL)
