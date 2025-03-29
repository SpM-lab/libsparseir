#pragma once

#include "sparseir/kernel.hpp"

namespace sparseir {

// Function to compute matrix from Gauss rules
template <typename T>
Eigen::MatrixX<T> matrix_from_gauss(const AbstractKernel &kernel, const Rule<T> &gauss_x,
                                    const Rule<T> &gauss_y)
{

    size_t n = gauss_x.x.size();
    size_t m = gauss_y.x.size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> res(n, m);

    // Parallelize using threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < n; ++i) {
        threads.emplace_back([&, i]() {
            for (size_t j = 0; j < m; ++j) {
                res(i, j) =
                    kernel.compute(gauss_x.x[i], gauss_y.x[j],
                                   gauss_x.x_forward[i], gauss_x.x_backward[i]);
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    return res;
}

// Specialization of matrix_from_gauss for std::shared_ptr<AbstractKernel>
/*
template <typename T>
Eigen::MatrixX<T>
matrix_from_gauss(const std::shared_ptr<AbstractKernel> &kernel,
                  const Rule<T> &gauss_x, const Rule<T> &gauss_y)
{
    size_t n = gauss_x.x.size();
    size_t m = gauss_y.x.size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> res(n, m);

    // Parallelize using threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < n; ++i) {
        threads.emplace_back([&, i]() {
            for (size_t j = 0; j < m; ++j) {
                res(i, j) = kernel->compute(gauss_x.x[i], gauss_y.x[j],
                                            gauss_x.x_forward[i],
                                            gauss_x.x_backward[i]);
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    return res;
}
*/

// Function to validate symmetry and extract the right-hand side of the segments
template <typename T>
std::vector<T> symm_segments(const std::vector<T> &x)
{
    using std::abs;
    // Check if the vector x is symmetric
    for (size_t i = 0, n = x.size(); i < n / 2; ++i) {
        if (abs(x[i] + x[n - i - 1]) > std::numeric_limits<T>::epsilon()) {
            throw std::runtime_error("segments must be symmetric");
        }
    }

    // Extract the second half of the vector starting from the middle
    size_t mid = x.size() / 2;
    std::vector<T> xpos(x.begin() + mid, x.end());

    // Ensure the first element of xpos is zero; if not, prepend zero
    if (xpos.empty() || abs(xpos[0]) > std::numeric_limits<T>::epsilon()) {
        xpos.insert(xpos.begin(), T(0));
    }

    return xpos;
}

} // namespace sparseir