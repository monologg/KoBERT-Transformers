.PHONY: style quality test

check_dirs := kobert_transformers/ tests/

style:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

test:
	pytest