.PHONY: install install-dev test check smoke

install:
	python -m pip install -e .[render]

install-dev:
	python -m pip install -e .[render,dev]

test:
	python -m pytest

check:
	python -m ruff check .

smoke:
	python scripts/smoke_test.py
