#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <numeric>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>

namespace sparseir {

namespace util {

// Implementation of make_unique for C++11
// This will be used when std::make_unique is not available
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace util


// Type trait to check if a type is a std::shared_ptr
template <typename T>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};


// julia> sort = sortperm(s; rev=true)
// Implement sortperm in C++
std::vector<size_t> sortperm_rev(const Eigen::VectorXd &vec);

template <typename Container>
bool issorted(const Container &c)
{
    if (c.size() <= 1) {
        return true;
    }

    return std::is_sorted(c.begin(), c.end());
}

// Overload for Eigen vectors
template <typename Derived>
bool issorted(const Eigen::MatrixBase<Derived> &vec)
{
    if (vec.size() <= 1) {
        return true;
    }

    for (Eigen::Index i = 1; i < vec.size(); ++i) {
        if (vec[i] < vec[i - 1]) {
            return false;
        }
    }
    return true;
}

template <typename T, int N>
bool tensorIsApprox(const Eigen::Tensor<T, N> &a, const Eigen::Tensor<T, N> &b,
                    double tol = 1e-12)
{
    // Ensure dimensions match (assuming dimensions() returns an
    // Eigen::DSizes<>)
    if (a.dimensions() != b.dimensions()) {
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        // Compare the absolute difference of each complex number.
        if (std::abs(a.data()[i] - b.data()[i]) > tol) {
            return false;
        }
    }
    return true;
}

template <typename T>
inline std::vector<T> diff(const std::vector<T> &xs)
{
    std::vector<T> diff(xs.size() - 1);
    for (size_t i = 0; i < xs.size() - 1; ++i) {
        diff[i] = xs[i + 1] - xs[i];
    }
    return diff;
}

template <typename T>
inline Eigen::VectorX<T> diff(const Eigen::VectorX<T> &xs)
{
    Eigen::VectorX<T> diff(xs.size() - 1);
    for (Eigen::Index i = 0; i < xs.size() - 1; ++i) {
        diff[i] = xs[i + 1] - xs[i];
    }
    return diff;
}

} // namespace sparseir