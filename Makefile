
PYTHON = .venv/bin/python
PIP = .venv/bin/pip
VENV = .venv

.PHONY: all venv install run run-mobilenet evaluate test clean help

all: install

# Create the virtual environment if it doesn't exist
$(PYTHON):
	python3 -m venv $(VENV)

venv: $(PYTHON)

# Install project dependencies
install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

# Run live object detection with YOLOv8
run:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) src/main.py --model yolo

# Run live object detection with MobileNet SSD
run-mobilenet:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) src/main.py --model mobilenet

# Compare YOLOv8 and MobileNet
evaluate:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) src/main.py --model compare --frames 50

# Run available unit tests
test:
	$(PYTHON) -m unittest discover -s test -p "test_*.py" -v

# Remove locally generated environment and caches
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make install        Create environment and install dependencies"
	@echo "  make run            Run YOLOv8 detection"
	@echo "  make run-mobilenet  Run MobileNet detection"
	@echo "  make evaluate       Compare both models"
	@echo "  make test           Run unit tests"
	@echo "  make clean          Remove environment and Python caches"
