#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <array>
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

template <int N>
Eigen::array<int, N> getperm(int src, int dst) {
    Eigen::array<int, N> perm;
    if (src == dst) {
        for (int i = 0; i < N; ++i) {
            perm[i] = i;
        }
        return perm;
    }

    int pos = 0;
    for (int i = 0; i < N; ++i) {
        if (i == dst) {
            perm[i] = src;
        } else {
            // src の位置をスキップ
            if (pos == src)
                ++pos;
            perm[i] = pos;
            ++pos;
        }
    }
    return perm;
}

// movedim: テンソル arr の次元 src を次元 dst に移動する（他の次元の順序はそのまま）
template<typename T, int N>
Eigen::Tensor<T, N> movedim(const Eigen::Tensor<T, N>& arr, int src, int dst) {
    if (src == dst) {
        return arr;
    }
    auto perm = getperm<N>(src, dst);
    return arr.shuffle(perm);
}

}