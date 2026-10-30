UV ?= uv
PYTHON := .venv/bin/python
WORKSPACE ?= .local/real
PREFECT_API_URL ?= http://127.0.0.1:4208/api
export PREFECT_API_URL
export PREFECT_SERVER_ANALYTICS_ENABLED := false
export TOKENIZERS_PARALLELISM := false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO := 0.5
export PYTORCH_MPS_LOW_WATERMARK_RATIO := 0.4

.PHONY: install install-model check check-core check-negative check-extra lint prefect fetch-model export-model release serve benchmark smoke stub-experiment release-audit
install:
	$(UV) sync --frozen
install-model:
	$(UV) sync --frozen --extra model
check:
	$(PYTHON) scripts/check_all.py
check-core:
	$(PYTHON) -m unittest discover -s tests -v
check-negative:
	$(PYTHON) scripts/mutation_check.py
check-extra:
	$(PYTHON) scripts/negative_contracts.py
lint:
	.venv/bin/ruff check --no-cache release.py lab.py publication.py artifacts.py binding.py export_existing.py pipeline.py backends.py service.py scripts checks checks_extra
	.venv/bin/ruff format --check release.py lab.py publication.py artifacts.py binding.py export_existing.py pipeline.py backends.py service.py scripts checks checks_extra
prefect:
	.venv/bin/prefect server start --host 127.0.0.1 --port 4208
fetch-model:
	$(PYTHON) scripts/fetch_model.py
export-model:
	$(PYTHON) export_existing.py --workspace "$(WORKSPACE)" --source-dir .local/model-source/adapter
release:
	$(PYTHON) lab.py --workspace "$(WORKSPACE)" run --manifest manifest-real.json --report release-report.json
serve:
	$(PYTHON) service.py --workspace "$(WORKSPACE)" --port 8098
benchmark:
	$(PYTHON) scripts/benchmark.py
smoke:
	$(PYTHON) scripts/real_smoke.py --workspace "$(WORKSPACE)"
stub-experiment:
	$(PYTHON) scripts/stub_experiment.py --workspace .local/stub-experiment
release-audit:
	$(PYTHON) scripts/release_audit.py --workspace "$(WORKSPACE)"
