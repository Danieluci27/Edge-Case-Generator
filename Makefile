PYTHON ?= python3

.PHONY: install test train-debug smoke export collect-smoke collect-pipeline

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest

train-debug:
	$(PYTHON) -m edge_case_generator.scripts.train_edge_case_generator --config edge_case_generator/configs/debug.yaml

smoke:
	$(PYTHON) -m edge_case_generator.scripts.run_smoke_test

export:
	$(PYTHON) -m edge_case_generator.scripts.export_augmented_dataset --config edge_case_generator/configs/debug.yaml

collect-smoke:
	$(PYTHON) -m edge_case_generator.scripts.run_data_collection_smoke_test --config edge_case_generator/configs/collection_debug.yaml

collect-pipeline:
	$(PYTHON) -m edge_case_generator.scripts.run_data_collection_pipeline --config edge_case_generator/configs/collection_default.yaml
