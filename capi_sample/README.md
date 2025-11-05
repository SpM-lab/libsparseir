# Samples in C

## List of samples

- `sample1.c`: how to create a fermionic finite temperature basis with the logistic kernel.
- `sample2.c`: how to create fermionic and bosonic finite temperature bases with the logistic kernel, and a bosonic basis with the regularized kernel.
- `sample3.c`: how to create sparse grids in the Matsubara-frequency and imaginary-time domains.



## Build and run

### Option 1: Build with pre-installed library

After installing the library, you can build the samples as follows:

```bash
# set the path to the libsparseir installation
export SparseIR_DIR=$HOME/opt/libsparseir/share/cmake
cmake -S . -B ./build
cmake --build build --target test
```

### Option 2: Build C++ backend and samples together

You can also build the C++ backend and run the samples in one go:

```bash
./run_sample.sh
```

This script will:
1. Build the C++ backend with BLAS support in `work_cxx/` directory
2. Install the backend to the `install/` subdirectory
3. Build and run all sample programs