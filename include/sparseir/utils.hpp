#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <vector>

namespace sparseir {
// julia> sort = sortperm(s; rev=true)
// Implement sortperm in C++
inline std::vector<size_t> sortperm_rev(const Eigen::VectorXd &vec)
{
    std::vector<size_t> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                [&vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; });
        return indices;
}

template<typename Container>
bool issorted(const Container& c) {
    if (c.size() <= 1) {
        return true;
    }

    return std::is_sorted(c.begin(), c.end());
}

// Overload for Eigen vectors
template<typename Derived>
bool issorted(const Eigen::MatrixBase<Derived>& vec) {
    if (vec.size() <= 1) {
        return true;
    }

    for (Eigen::Index i = 1; i < vec.size(); ++i) {
        if (vec[i] < vec[i-1]) {
            return false;
        }
    }
    return true;
}

}