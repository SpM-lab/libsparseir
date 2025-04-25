# libsparseir Python Package

Python interface for the libsparseir library, providing tools for sparse intermediate representation (IR) calculations.

## Installation

Install `uv` via:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

See [installation-methods](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) to learn more.

```bash
# For development
git clone https://github.com/yourusername/libsparseir.git
cd libsparseir/python
uv sync
source .venv/bin/activate
```

## Features

- Kernel functions for IR calculations
- Finite temperature basis functions
- Sampling methods for fermionic and bosonic systems
- DLR (Discrete Lehmann Representation) support
- Efficient memory management

## Development

### Testing

```bash
# Create virtual environment
uv sync
source .venv/bin/activate
pytest
```

or

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=libsparseir
```

### Code Quality

```bash
# Format code
uv run ruff format

# Check code quality
uv run ruff check
```
