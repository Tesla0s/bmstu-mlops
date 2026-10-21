export HF_HOME ?= $(CURDIR)/.cache/huggingface
export TOKENIZERS_PARALLELISM := false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO := 0.5
export PYTORCH_MPS_LOW_WATERMARK_RATIO := 0.4

.PHONY: install up down sweep promote predict check check-storage
install:
	uv sync --frozen
up:
	docker compose -p mlops-hw6 up --build -d --wait
down:
	docker compose -p mlops-hw6 down
sweep:
	uv run python -m src.sweep
promote:
	uv run python -m evaluation.cli promote
predict:
	uv run python -m src.predict --uri models:/github-issue-qwen-adapter@champion --prompt "Title: Editor crashes on save. Description: Ctrl+S closes the editor."
check:
	bash tests/check.sh
check-storage:
	uv run python tests/check_storage.py

.PHONY: report candidate baseline-init evaluate eval check-eval check-negative check-all baseline-update predict-candidate
report:
	uv run python -m src.report
candidate:
	uv run python -m evaluation.cli candidate
baseline-init:
	uv run python -m evaluation.cli baseline-init
evaluate:
	uv run python -m evaluation.cli evaluate
eval:
	uv run python -m evaluation.cli gate
check-eval:
	uv run python -m unittest discover -s tests -p 'test_*.py' -v
check-negative:
	uv run python -m evaluation.cli negative
check-all: check check-eval
baseline-update:
	uv run python -m evaluation.cli baseline-update
predict-candidate:
	uv run python -m src.predict --uri models:/github-issue-qwen-adapter@candidate --prompt "Title: Add CSV export. Description: Please add a new export command."
