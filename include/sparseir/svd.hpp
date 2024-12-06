#pragma once

// C++ Standard Library headers
#include <stdexcept>
#include <iostream>

// Eigen headers
#include <Eigen/Dense>

namespace sparseir
{

    using namespace Eigen;

    template <typename T>
    std::tuple<MatrixX<T>, VectorX<T>, MatrixX<T>> compute_svd(const MatrixX<T> &A, int n_sv_hint = 0, std::string strategy = "default")
    {
        if (n_sv_hint != 0)
        {
            std::cout << "n_sv_hint is set but will not be used in the current implementation!" << std::endl;
        }

        if (strategy != "default")
        {
            std::cout << "strategy is set but will not be used in the current implementation!" << std::endl;
        }

        MatrixX<T> A_copy = A; // create a copy of A
        return tsvd<T>(A_copy);
        // auto svd_result = tsvd<T>(A_copy);
        // MatrixX<T> u = std::get<0>(svd_result);
        // VectorX<T> s = std::get<1>(svd_result);
        // MatrixX<T> v = std::get<2>(svd_result);

        // return std::make_tuple(u, s, v);
    }
}
