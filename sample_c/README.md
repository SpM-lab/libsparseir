# libsparseir samples

## List of samples

- `sample1.c`: A simple sample that demonstrates how to create a fermionic finite temperature basis with the logistic kernel.
- `sample2.c`: A sample that demonstrates how to create fermionic and bosonic finite temperature bases with the logistic kernel, and a bosonic basis with the regularized kernel.
- `sample3.c`: A sample that demonstrates how to create sparse grids in the Matsubara-frequency and imaginary-time domains.



## Build and run
After installing the library, you can build the samples as follows:

```bash
# set the path to the libsparseir installation
export SparseIR_DIR=$HOME/opt/libsparseir/share/cmake
cmake --build build
cmake --build build --target test
```