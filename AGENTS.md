# AGENTS.md - AI Assistant Workflow for libsparseir

This file provides guidance for AI assistants working on the libsparseir project, particularly for the Python interface.

## Python Interface Build Process

The Python interface (`libsparseir/python/`) uses `scikit-build-core` with CMake to build the C++ backend and generate Python bindings.

### Prerequisites

1. **uv** - Python package manager (REQUIRED)
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or on macOS:
   brew install uv
   ```

2. **clang (libclang)** - Required for auto-generating ctypes bindings
   ```bash
   # Install clang Python package via uv
   cd libsparseir/python
   uv sync --group dev
   
   # On macOS, also install llvm for libclang:
   brew install llvm
   # Set DYLD_LIBRARY_PATH if needed (see README.md)
   ```

3. **CMake** (>= 3.25) - Build system
4. **C++ compiler** with C++11 support

### Build Process

The build process is automated through shell scripts:

#### Option 1: Using `run_tests.sh` (Recommended for testing)

This script performs a clean build and runs tests:

```bash
cd libsparseir/python
bash run_tests.sh
```

What it does:
1. Cleans up previous build artifacts (`.venv`, `_skbuild`, `dist`, etc.)
2. Runs `setup_build.py` to copy source files
3. Runs `uv sync --refresh` to build the package (CMake will auto-generate `ctypes_autogen.py`)
4. Runs `uv run pytest tests/ -v` to execute tests

#### Option 2: Manual build steps

```bash
cd libsparseir/python

# 1. Prepare build environment
python3 setup_build.py

# 2. Build and install
uv sync --refresh

# 3. Run tests
uv run pytest tests/ -v
```

### CMake Build Integration

The CMake build process (`CMakeLists.txt`) automatically:

1. **Requires `uv`** - The build will fail if `uv` is not found
2. **Requires `clang`** - The build will fail if `clang` Python package is not available
3. **Auto-generates `ctypes_autogen.py`** - The `generate_ctypes` target runs `tools/gen_ctypes.py` before building the library
4. **Installs generated file** - `pylibsparseir/ctypes_autogen.py` is installed as part of the package

The generation happens via:
- Custom target: `generate_ctypes`
- Dependency: `sparseir` library depends on `generate_ctypes`
- Command: `uv run python tools/gen_ctypes.py`

### Updating ctypes Bindings

If you modify the C-API header (`sparseir.h`), you need to regenerate the bindings:

```bash
cd libsparseir/python
bash update_wrapper.sh
```

This script:
1. Checks for header files (runs `setup_build.py` if needed)
2. Sets up `libclang` library path for macOS
3. Runs `uv run python tools/gen_ctypes.py`
4. Verifies the generated file

**Note:** The bindings are automatically regenerated during CMake builds, so manual regeneration is only needed if you want to update them outside of a build.

### Key Files

- **`tools/gen_ctypes.py`** - Script to parse `sparseir.h` and generate `ctypes_autogen.py`
- **`pylibsparseir/ctypes_autogen.py`** - Auto-generated ctypes bindings (DO NOT EDIT MANUALLY)
- **`pylibsparseir/core.py`** - Python wrapper that uses the generated bindings
- **`CMakeLists.txt`** - CMake configuration that enforces generation during build
- **`run_tests.sh`** - Complete build and test script
- **`update_wrapper.sh`** - Script to manually regenerate bindings

### Troubleshooting

1. **`uv` not found**: Install `uv` using the commands above
2. **`clang` not found**: Run `uv sync --group dev` to install development dependencies
3. **`libclang.dylib` not found** (macOS): Install `llvm` via Homebrew and set `DYLD_LIBRARY_PATH`
4. **`ctypes_autogen.py` missing**: The file should be auto-generated during build. If missing, run `bash update_wrapper.sh`

### Development Workflow

1. Make changes to C-API (`backend/cxx/include/sparseir/sparseir.h`)
2. Run `bash update_wrapper.sh` to regenerate bindings (or let CMake do it)
3. Update Python wrapper code if needed (`pylibsparseir/core.py`)
4. Run `bash run_tests.sh` to build and test
5. Commit both source changes and `ctypes_autogen.py` (the generated file should be committed)

### Important Notes

- **Never edit `ctypes_autogen.py` manually** - It is auto-generated and will be overwritten
- **Always commit `ctypes_autogen.py`** - This ensures the package can be built even without `clang`
- **Use `uv` for all Python operations** - This ensures consistent environment
- **CMake build requires both `uv` and `clang`** - The build will fail if either is missing

