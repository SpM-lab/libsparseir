# libsparseir Python Bindings

Python bindings for the libsparseir library using CFFI.

## Prerequisites

Before installing the Python bindings, make sure you have:
1. Build `libsparseir`. Namely run the following command:

```sh
$ cd .. && bash build_capi.sh && cd -
```

2. Install `uv` package manager

```sh
$ uv python install 3.12
$ uv sync
```

1. First, build the CFFI bindings:
```bash
uv run build_cffi.py
```

2. Run the tests:
```bash
uv run pytest tests/
```

## Cleanup

To clean up build artifacts and temporary files:

```bash
rm -rf src/libsparseir/__pycache__ tests/__pycache__ .pytest_cache .ruff_cache
rm -rf src/libsparseir/_libsparseir_cffi*
```
