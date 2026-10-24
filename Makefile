.PHONY: check
check:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .
	uv run --no-sync mypy
	uv run --no-sync pyright
	uv run --no-sync pytest -q --cov --cov-fail-under=95
	uv build -q && uvx twine@7.0.0 check --strict dist/*
