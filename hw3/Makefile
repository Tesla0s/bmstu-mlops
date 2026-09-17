UV ?= uv

.NOTPARALLEL: verify

.PHONY: install fetch repro v1 v2 diff dag diversity contamination check test lint verify push-data pull-data

install:
	$(UV) sync --frozen

# fetch создаёт новую выгрузку; pull-data восстанавливает зафиксированную.
fetch:
	$(UV) run python scripts/fetch_source.py
	$(UV) run dvc commit --force source.dvc

repro:
	$(UV) run dvc repro

v1:
	$(UV) run python scripts/set_version.py v1
	$(UV) run dvc repro

v2:
	$(UV) run python scripts/set_version.py v2
	$(UV) run dvc repro

diff:
	$(UV) run dvc metrics diff hw3-v1 hw3-v2 --targets metrics/clean.json

dag:
	$(UV) run dvc dag

diversity:
	$(UV) run python -m src.diversity

contamination:
	$(UV) run python scripts/check_contamination.py

# Проверка преподавателя сохранена без изменений.
check:
	bash tests/check.sh

test:
	$(UV) run python -m unittest discover -s tests -p 'test_*.py' -v

lint:
	$(UV) run ruff check src scripts tests
	$(UV) run ruff format --check src scripts tests

verify: repro test lint check

push-data:
	$(UV) run dvc push

pull-data:
	$(UV) run dvc pull
