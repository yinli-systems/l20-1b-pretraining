PYTHON ?= python3

.PHONY: verify test language-ci
verify:
	$(PYTHON) tools/verify_repository.py
	$(PYTHON) tools/build_language_showcase.py --check
	$(PYTHON) pretraining/reproducibility/recompute.py

test:
	$(PYTHON) -m pytest -q

language-ci:
	$(PYTHON) pretraining/reproducibility/recompute_efficiency_ci.py
