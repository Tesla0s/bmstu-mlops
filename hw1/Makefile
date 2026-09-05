.PHONY: install generate bench check test verify clean

# Инструменты и скачанные модели остаются внутри проекта.
export PATH := $(CURDIR)/.tools/bin:$(PATH)
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.tools/python
export HF_HOME := $(CURDIR)/.cache/huggingface
export TOKENIZERS_PARALLELISM := false

UV := $(if $(wildcard $(CURDIR)/.tools/bin/uv),$(CURDIR)/.tools/bin/uv,uv)

install:
	$(UV) sync --locked

generate:
	$(UV) run --locked python -m src.generate

bench:
	$(UV) run --locked python -m src.bench

check:
	bash tests/check.sh

test:
	$(UV) run --locked python -m unittest discover -s tests -v

verify:
	$(UV) run --locked python -m tests.verify_runtime

clean:
	rm -f out1.txt out2.txt params.yaml.bak params.yaml.orig
