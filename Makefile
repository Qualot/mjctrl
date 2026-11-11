SHELL := /bin/bash

.PHONY: help format test
.DEFAULT: help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@echo "  help: Show this help"
	@echo "  format: Run code formatting and linting"
	@echo "  test: Run all tests"

format:
	uv run black .
	uv run ruff check --fix .

test:
	uv run pytest -n auto