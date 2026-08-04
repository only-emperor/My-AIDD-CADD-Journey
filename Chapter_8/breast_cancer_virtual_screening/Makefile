.PHONY: install install-full test lint compile init demo-help clean

install:
	python -m pip install -e .

install-full:
	python -m pip install -e ".[admet,parquet,dev]"

test:
	pytest

lint:
	ruff check src tests

compile:
	python -m compileall -q src tests

init:
	bcvs init project

demo-help:
	bcvs --help
	bcvs screen --help

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
