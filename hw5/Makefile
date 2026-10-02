export HF_HOME ?= $(CURDIR)/.cache/huggingface
export HF_HUB_OFFLINE ?= 1
export TRANSFORMERS_VERBOSITY = error
export HF_HUB_VERBOSITY = error
export TOKENIZERS_PARALLELISM = false
export PYTHONHASHSEED = 42
export PYTHONUNBUFFERED = 1

.PHONY: install train train-all train-freeze compare plot check test lint audit push-data pull-data
install:
	uv sync --frozen
train:
	uv run --locked dvc repro train_all train_freeze curves report
train-all:
	uv run --locked dvc repro train_all
train-freeze:
	uv run --locked dvc repro train_freeze
compare:
	uv run --locked dvc repro compare
plot:
	uv run --locked python -m src.plot
check:
	bash tests/check.sh
test:
	uv run --locked python -m unittest discover -s tests -p 'test_*.py' -v
lint:
	uv run --locked ruff check src scripts tests
	uv run --locked ruff format --check src scripts tests
audit:
	uv run --locked dvc repro audit
push-data:
	uv run --locked dvc push
pull-data:
	uv run --locked dvc pull

.PHONY: report
report:
	uv run --locked dvc repro report
