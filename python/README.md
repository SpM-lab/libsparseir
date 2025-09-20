# Python bindings for libsparseir

This is a low-level binding for the [libsparseir](https://github.com/SpM-lab/libsparseir) library.

## Requirements

- Python >= 3.10
- CMake (for building the C++ library)
- C++11 compatible compiler
- numpy

### Optional Dependencies

- **OpenBLAS** (recommended for better performance)
  - macOS: `brew install openblas`
  - Ubuntu/Debian: `sudo apt install libopenblas-dev`
  - CentOS/RHEL: `sudo yum install openblas-devel`

## Build

### Install Dependencies and Build

```bash
# First, prepare the build by copying necessary files from parent directory
python3 prepare_build.py

# Then build the package
uv build
```

This will:
- Copy source files (`src/`, `include/`, `cmake/`) from the parent libsparseir directory
- Build the C++ libsparseir library using CMake with BLAS support
- Create both source distribution (sdist) and wheel packages

### Development Build

For development, you can also use:

```bash
# Prepare build files
python3 prepare_build.py

# Install in development mode
uv sync
```

### Build with OpenBLAS Support

OpenBLAS support is enabled by default in the build configuration. The build system will automatically detect OpenBLAS if it's installed in standard locations.

If OpenBLAS is installed in a custom location, you may need to set additional environment variables:

```bash
export CMAKE_PREFIX_PATH="/path/to/openblas"
python3 prepare_build.py
uv build
```

### Clean Build Artifacts

To remove build artifacts and files copied from the parent directory:

```bash
uv run clean
```

This will remove:
- Build directories: `build/`, `dist/`, `*.egg-info`
- Copied source files: `include/`, `src/`, `cmake/` (copied by `prepare_build.py`)
- Compiled libraries: `pylibsparseir/*.so`, `pylibsparseir/*.dylib`, `pylibsparseir/*.dll`
- Cache directories: `pylibsparseir/__pycache__`

### Build Process Overview

The build process works as follows:

1. **File Preparation**: `prepare_build.py` copies necessary files from the parent libsparseir directory:
   - Source files (`../src/` → `src/`)
   - Header files (`../include/` → `include/`)
   - CMake configuration (`../cmake/` → `cmake/`)

2. **Package Building**: `uv build` or `uv sync` uses scikit-build-core to:
   - Configure CMake with BLAS support enabled
   - Compile the C++ library with dynamic BLAS symbol lookup (for NumPy compatibility)
   - Package everything into distributable wheels and source distributions

3. **Installation**: The built package includes the compiled shared library and Python bindings

**Why File Copying?**: The `prepare_build.py` script copies files from the parent directory instead of using symbolic links to ensure:
- Cross-platform compatibility (Windows doesn't handle symlinks well)
- Proper inclusion in source distributions (sdist)
- Clean separation between the main C++ library and Python bindings

## Performance Notes

### BLAS Support

This package supports BLAS libraries for improved linear algebra performance:

- **With OpenBLAS**: Significant performance improvements for matrix operations
- **Without BLAS**: Uses Eigen's built-in implementations (still efficient, but slower for large matrices)

The build system will automatically detect and use OpenBLAS if available. You can verify BLAS support by checking the build output for messages like:

```
BLAS support enabled
Found OpenBLAS at: /opt/homebrew/opt/openblas
```

### Troubleshooting

**Build fails with missing source files:**
```bash
# Make sure to run prepare_build.py first
python3 prepare_build.py
uv build
```

**Build fails with "Could NOT find BLAS":**
```bash
# Install OpenBLAS first
brew install openblas  # macOS
sudo apt install libopenblas-dev  # Ubuntu

# Then build with proper CMake path
export CMAKE_PREFIX_PATH="/path/to/openblas"
python3 prepare_build.py
uv build
```

**OpenBLAS not detected automatically:**
```bash
# Set CMake prefix path manually
export CMAKE_PREFIX_PATH="/usr/local/opt/openblas"  # or your OpenBLAS path
python3 prepare_build.py
uv build
```

**Clean rebuild:**
```bash
# Remove all copied files and build artifacts
uv run clean
python3 prepare_build.py
uv build
```

**Verify BLAS support in built package:**
```python
import pylibsparseir
# Check build logs for "BLAS support enabled" message
# BLAS symbols are resolved dynamically through NumPy at runtime
```
