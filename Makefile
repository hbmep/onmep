SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/python -m pip
PIP_NO_CACHE := --no-cache-dir

.PHONY: base env env-cuda12 dev dev-cuda12 clean-docs docs

base:
	rm -rf $(VENV) build
	@echo "Creating virtual environment with $(PY)..."
	$(PY) -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip

env: base
	@echo "Installing package..."
	$(PIP) install $(PIP_NO_CACHE) .

env-cuda12: base
	@echo "Installing package..."
	$(PIP) install $(PIP_NO_CACHE) ".[cuda12]"

dev: base
	@echo "Installing package for development..."
	$(PIP) install $(PIP_NO_CACHE) -e ".[dev]"
	$(PIP) install $(PIP_NO_CACHE) -e ../hbmep

dev-cuda12: base
	@echo "Installing package for development..."
	$(PIP) install $(PIP_NO_CACHE) -e ".[dev,cuda12]"
	$(PIP) install $(PIP_NO_CACHE) -e ../hbmep
