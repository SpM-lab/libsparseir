# libsparseir

> [!WARNING]
> This C++ project is still work in progress Please use other repositories such as
> - https://github.com/SpM-lab/sparse-ir
> - https://github.com/SpM-lab/SparseIR.jl
> - https://github.com/SpM-lab/sparse-ir-fortran

## Description

This C++ library provides routines for constructing and working with the intermediate representation of correlation functions. It provides:

- on-the-fly computation of basis functions for arbitrary cutoff Λ
- basis functions and singular values are accurate to full precision
- routines for sparse sampling

We use [tuwien-cms/libxprec](https://github.com/tuwien-cms/libxprec) as a double-double precision arithmetic library.

## Building and Testing

Just run:

```sh
$ rm -rf ./build && cmake -S . -B ./build -DSPARSEIR_BUILD_TESTING=ON && cmake --build ./build -j && ./build/test/libsparseirtests
```
