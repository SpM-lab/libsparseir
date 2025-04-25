# libsparseir Python Package

Python interface for the libsparseir library, providing tools for sparse intermediate representation (IR) calculations.

## Installation

```bash
# Install the package
pip install libsparseir

# For development
git clone https://github.com/yourusername/libsparseir.git
cd libsparseir/python
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Features

- Kernel functions for IR calculations
- Finite temperature basis functions
- Sampling methods for fermionic and bosonic systems
- DLR (Discrete Lehmann Representation) support
- Efficient memory management

## Usage

```python
import numpy as np
from libsparseir import (
    Kernel,
    FermionicFiniteTempBasis,
    spir_order_type
)

# Create a kernel
kernel = Kernel.logistic(lambda_=1.0)

# Create a finite temperature basis
basis = FermionicFiniteTempBasis.new(
    beta=10.0,      # Inverse temperature
    omega_max=1.0,  # Maximum frequency
    epsilon=1e-6    # Accuracy parameter
)

# Create sampling objects
tau_sampling = basis.create_tau_sampling()
matsubara_sampling = basis.create_matsubara_sampling()

# Create DLR object
dlr = basis.create_dlr()

# Evaluate functions
x = np.linspace(0, 1, 100)
output = sampling.evaluate_dd(x, order=spir_order_type.SPIR_ORDER_ROW_MAJOR)
```

## Development

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Testing

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
