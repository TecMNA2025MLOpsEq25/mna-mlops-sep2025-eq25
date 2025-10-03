#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Publicmna-mlops-sep2025-eq25
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 Publicmlops
	isort --check --diff Publicmlops
	black --check Publicmlops

## Format source code with black
.PHONY: format
format:
	isort Publicmlops
	black Publicmlops



## Run tests
.PHONY: test
test:
	python -m pytest tests
## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	az storage blob download-batch -s dvc-data-remote/data/ \
		-d data/
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	az storage blob upload-batch -d dvc-data-remote/data/ \
		-s data/
	



## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
