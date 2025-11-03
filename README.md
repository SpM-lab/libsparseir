# libsparseir
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SpM-lab/libsparseir)
[![CMake on a single platform](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml/badge.svg)](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml)
[![Create Tag and Release](https://github.com/SpM-lab/libsparseir/actions/workflows/CreateTag.yml/badge.svg)](https://github.com/SpM-lab/libsparseir/actions/workflows/CreateTag.yml)

> [!WARNING]
> This C++ project is still under construction. Please use other repositories:
> - https://github.com/SpM-lab/sparse-ir
> - https://github.com/SpM-lab/SparseIR.jl
> - https://github.com/SpM-lab/sparse-ir-fortran

## Description

This library provides functions for constructing and working with the intermediate representation of correlation functions. It provides:

- on-the-fly computation of basis functions for arbitrary cutoff Λ
- basis functions and singular values are accurate to full precision
- functions for sparse sampling

The library provides a C-API and associated thin Julia, Python, and Fortran bindings.

For user-friendly Python and Julia interfaces, please refer to the following repositories:
- https://github.com/SpM-lab/sparse-ir
- https://github.com/SpM-lab/SparseIR.jl

Currently, the library is implemented in C++11. A Rust backend with a compatible C-API is under development.

## Building and Installation
See [backend/cxx/README.md](backend/cxx/README.md) for more details on building the C++ backend.

For building the Fortran bindings, see [fortran/test_with_cxx_backend.sh](fortran/test_with_cxx_backend.sh).

For bulding the Python bindings, see [python/run_tests.sh](python/run_tests.sh).

```c
void spir_register_dgemm_zgemm_ilp64(void* dgemm_fn, void* zgemm_fn);
```
## BLAS Support
For BLAS support, refer to the [backend/cxx/README.md](backend/cxx/README.md) for more details.

## Sample code in C
The directory [`./capi_sample/`](./capi_sample/) contains sample code in C.
The shell script [`./capi_sample/run_sample.sh`](./capi_sample/run_sample.sh) will build the C++ backend and run the samples.

## For developers

This project uses GitHub Actions for continuous integration and automated releases.

### Automated Release Process

The release process is fully automated:

1. Update version numbers using the provided script:
   ```bash
   python update_version.py 0.4.3
   ```
   This automatically updates:
   - `backend/cxx/include/sparseir/version.h` (C++ library version)
   - `python/pyproject.toml` (Python package version)

2. Review and commit the changes:
   ```bash
   git diff  # Review changes
   git add -A
   git commit -m "Bump version to 0.4.3"
   ```

3. Push changes to the main branch

4. The GitHub Action will automatically:
   - Extract the version from the header file
   - Check if a tag with that version already exists
   - Create a new tag and release if the version is new
   - Generate release notes automatically
   - Build and publish Python packages to PyPI and conda-forge

#### Version Update Script Usage

The `update_version.py` script provides a convenient way to update versions across all components:

```bash
# Show current versions
python update_version.py

# Update to a new version
python update_version.py 1.0.0

# The script validates version format (x.y.z)
python update_version.py 1.0  # Error: Invalid format
```