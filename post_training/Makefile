SHELL := /bin/sh

PYTHON ?= python3
PACKAGE_DIR := dqs_preference_training_hf
TEST_PYTHONPATH := $(abspath $(PACKAGE_DIR)/src):$(abspath research/data_prep):$(abspath research/audits):$(abspath research/evaluation)
HF_DATASET_REPO := alwaysgood/dqs-post-training
HF_DATASET_REVISION := 33899900bbb516b5684c09d2e3fa71a192437f35
RELEASE_DIR ?= dist/dqs_preference_training_hf

.PHONY: test test-runtime test-research test-release test-package build-release

test: test-runtime test-research test-release test-package

test-runtime:
	PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m unittest discover -s tests/runtime -p 'test_*.py' -v

test-research:
	PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m unittest discover -s research/tests -p 'test_*.py' -v

test-release:
	PYTHONPATH="$(TEST_PYTHONPATH)" $(PYTHON) -m unittest discover -s tests/release -p 'test_*.py' -v

test-package:
	$(MAKE) -C $(PACKAGE_DIR) test PYTHON=$(PYTHON)

build-release:
	$(PYTHON) scripts/build_release.py \
		--output $(RELEASE_DIR) \
		--data-mode hf \
		--hf-repo-id $(HF_DATASET_REPO) \
		--hf-revision $(HF_DATASET_REVISION) \
		--replace
