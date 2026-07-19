# Makefile for Python project

# Variables
PYTHON := python3
PIP := pip3
LINT := pylint
TEST := pytest -vv

# Directories
SRC_DIR := muse
TEST_DIR := tests

# Targets
.PHONY: all clean lint test

all: lint test

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint:
	$(LINT) $(SRC_DIR)
	$(LINT) $(TEST_DIR)

test:
	$(TEST) $(TEST_DIR)

install:
	$(PIP) install -r requirements.txt

format:
	autopep8 --indent-size=2 --in-place --aggressive --aggressive ./**/*.py

# Add more targets as needed