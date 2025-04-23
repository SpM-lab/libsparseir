import numpy as np
from libsparseir import Kernel

def test_logistic_kernel():
    # Create a logistic kernel with lambda = 10.0
    kernel = Kernel.logistic(10.0)

    # Get the domain
    xmin, xmax, ymin, ymax = kernel.domain()
    assert xmin == -1.0
    assert xmax == 1.0
    assert ymin == -1.0
    assert ymax == 1.0

    # Create some test points
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)

    # Compute the kernel matrix
    K = kernel.matrix(x, y)
    assert K.shape == (10, 10)
    assert np.all(K >= 0)  # All values should be non-negative
    assert np.all(K <= 1)  # All values should be less than or equal to 1
