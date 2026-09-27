UV ?= uv
export HF_HOME ?= $(CURDIR)/.cache/huggingface
export TOKENIZERS_PARALLELISM := false
export HF_HUB_VERBOSITY := error
export TRANSFORMERS_VERBOSITY := error

.NOTPARALLEL: verify
.PHONY: install prepare sample tokenize repro check test lint verify audit push-data pull-data

install:
	$(UV) sync --frozen

prepare:
	$(UV) run --locked python -m scripts.prepare_data

# Совместимость с исходной проверкой: используется свой набор из ДЗ 3.
sample: prepare

tokenize:
	$(UV) run --locked python -m src.tokenize_data

repro:
	$(UV) run --locked dvc repro

check:
	bash tests/check.sh

test:
	$(UV) run --locked python -m unittest discover -s tests -p 'test_*.py' -v

lint:
	$(UV) run --locked ruff check src scripts tests
	$(UV) run --locked ruff format --check src scripts tests

audit:
	$(UV) run --locked python -m scripts.audit_artifacts

verify: repro test lint check audit

push-data:
	$(UV) run --locked dvc push

pull-data:
	$(UV) run --locked dvc pull
