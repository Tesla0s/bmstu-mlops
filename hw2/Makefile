.PHONY: install inspect check test clean
export PATH := $(CURDIR)/.tools/bin:$(PATH)
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export HF_HOME := $(CURDIR)/.cache/huggingface
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export TOKENIZERS_PARALLELISM := false
export HF_DEACTIVATE_ASYNC_LOAD := 1

UV := $(if $(wildcard $(CURDIR)/.tools/bin/uv),$(CURDIR)/.tools/bin/uv,uv)

install:
	$(UV) sync --locked
inspect:
	$(UV) run --locked python -m src.inspect_model
check:
	bash tests/check.sh
test:
	$(UV) run --locked python -m unittest discover -s tests -v
clean:
	rm -f docs/report.json
