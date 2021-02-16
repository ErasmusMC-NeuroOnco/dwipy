SHELL:=/usr/bin/env bash

.PHONY: lint
lint:
	poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	poetry run flake8 . --count --exit-zero --statistics

.PHONY: unit
unit:
	poetry run pytest

.PHONY: package
package:
	poetry check
	poetry run pip check

.PHONY: test
test: lint package unit
