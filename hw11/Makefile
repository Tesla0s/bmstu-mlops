PYTHON ?= .venv/bin/python
HW11_WORKSPACE ?= $(abspath ../../hw11/workspace)
export HW11_WORKSPACE

.PHONY: install lint experiments integration report check
install:
	uv venv .venv --python 3.12
	uv pip install --python $(PYTHON) --require-hashes -r requirements.lock -r analysis.lock
lint:
	"$(HW11_WORKSPACE)/bin/helm" lint chart
experiments:
	$(PYTHON) -u scripts/experiment_b.py
	$(PYTHON) scripts/recover_metrics.py
	$(PYTHON) -u scripts/experiment_a.py
	$(PYTHON) -u scripts/experiment_c.py
integration:
	$(PYTHON) -u scripts/test_integration.py
report:
	$(PYTHON) scripts/analyze.py
	$(PYTHON) scripts/build_report.py
check: lint
	$(PYTHON) scripts/analyze.py
	$(PYTHON) scripts/verify_evidence.py
