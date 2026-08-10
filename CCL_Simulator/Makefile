# =========================
# Config
# =========================
PYTHON      := python
BUILD_DIR   := build
VENDOR_DIR  := $(BUILD_DIR)/vendor
SIMCORE_PKG := simcore

DEPS := simpy networkx nuitka

# Nuitka flags
NUITKA_COMMON := --module $(SIMCORE_PKG) \
                 --follow-import-to=$(SIMCORE_PKG) \
                 --output-dir=$(BUILD_DIR)

# Debug build: safer / easier to iterate
NUITKA_DEBUG := --remove-output

# Release build: typical safe performance-oriented options
NUITKA_RELEASE := --remove-output \
                  --lto=yes \
                  --assume-yes-for-downloads

# =========================
# Phony targets
# =========================
.PHONY: all build debug release simcore simcore_debug simcore_release deps run clean

all: build

# Default build = release + deps
build: release
	@echo "Build finished."

debug: simcore_debug deps
	@echo "Debug build finished."

release: simcore_release deps
	@echo "Release build finished."

# Keep old target name for compatibility
simcore: simcore_release

# =========================
# Build simcore with Nuitka
# =========================
simcore_debug:
	@echo ">>> Building simcore (debug) with Nuitka"
	$(PYTHON) -m nuitka $(NUITKA_COMMON) $(NUITKA_DEBUG)

simcore_release:
	@echo ">>> Building simcore (release) with Nuitka"
	$(PYTHON) -m nuitka $(NUITKA_COMMON) $(NUITKA_RELEASE)

# =========================
# Vendor dependencies
# =========================
deps:
	@echo ">>> Vendoring dependencies: $(DEPS)"
	mkdir -p $(VENDOR_DIR)
	pip install --upgrade --target $(VENDOR_DIR) $(DEPS)

# =========================
# Run example with built core
# =========================
run:
	PYTHONPATH=$(BUILD_DIR):$(VENDOR_DIR) $(PYTHON) example_run.py

# =========================
# Clean
# =========================
clean:
	@echo ">>> Cleaning build artifacts"
	rm -rf $(BUILD_DIR)
	rm -rf $(SIMCORE_PKG)/*.so
	rm -rf $(SIMCORE_PKG)/__pycache__