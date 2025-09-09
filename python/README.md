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
uv sync
```

This will:
- Install Python dependencies (numpy)
- Build the C++ libsparseir library using CMake
- Install the Python package in development mode

### Build with OpenBLAS Support

To enable OpenBLAS support for improved performance:

```bash
# Set environment variable to enable BLAS
export SPARSEIR_USE_BLAS=1
uv sync
```

Or for a single build:

```bash
SPARSEIR_USE_BLAS=1 uv sync
```

The build system will automatically detect OpenBLAS if it's installed in standard locations. If OpenBLAS is installed in a custom location, you may need to set additional environment variables:

```bash
export CMAKE_PREFIX_PATH="/path/to/openblas"
export SPARSEIR_USE_BLAS=1
uv sync
```

### Clean Build Artifacts

To remove build artifacts and files copied from the parent directory:

```bash
uv run clean
```

This will remove:
- Build directories: `build/`, `dist/`, `*.egg-info`
- Copied source files: `include/`, `src/`, `fortran/`, `cmake/`, `CMakeLists.txt`
- Compiled libraries: `pylibsparseir/*.so`, `pylibsparseir/*.dylib`, `pylibsparseir/*.dll`
- Cache directories: `pylibsparseir/__pycache__`

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

**Build fails with "Could NOT find BLAS":**
```bash
# Install OpenBLAS first
brew install openblas  # macOS
sudo apt install libopenblas-dev  # Ubuntu

# Then force BLAS detection
SPARSEIR_USE_BLAS=1 uv sync
```

**OpenBLAS not detected automatically:**
```bash
# Set CMake prefix path manually
export CMAKE_PREFIX_PATH="/usr/local/opt/openblas"  # or your OpenBLAS path
export SPARSEIR_USE_BLAS=1
uv sync
```

**Verify BLAS support in built package:**
```python
import pylibsparseir
# Check build logs or library dependencies to confirm BLAS linking
```
