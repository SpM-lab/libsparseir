#pragma once

#include "sparseir/kernel.hpp"
#include <fstream>
#include <vector>
#include <algorithm>

namespace sparseir {

// Debug file output helper (shared with sve_impl.ipp)
inline std::ofstream& debug_out() {
    static std::ofstream debug_file("/tmp/sparseir_debug.log", std::ios::app);
    return debug_file;
}

// Function to compute matrix from Gauss rules
template <typename T>
Eigen::MatrixX<T> matrix_from_gauss(const AbstractKernel &kernel,
                                    const Rule<T> &gauss_x,
                                    const Rule<T> &gauss_y)
{
    size_t n = gauss_x.x.size();
    size_t m = gauss_y.x.size();
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> res(n, m);

    debug_out() << "[DEBUG C++] matrix_from_gauss: START, n=" << n << ", m=" << m << std::endl;
    debug_out().flush();

    // Optimize for FunctionKernel: use batch evaluation for entire 2D grid
    auto func_kernel = dynamic_cast<const FunctionKernel*>(&kernel);
    debug_out() << "[DEBUG C++] matrix_from_gauss: dynamic_cast result: " << (func_kernel ? "FunctionKernel found" : "NOT FunctionKernel") << std::endl;
    debug_out().flush();
    
    if (func_kernel) {
        debug_out() << "[DEBUG C++] matrix_from_gauss: FunctionKernel detected, n=" << n << ", m=" << m << std::endl;
        debug_out().flush();
        void* user_data = func_kernel->get_user_data();
        
        if constexpr (std::is_same<T, double>::value) {
            // Double precision: use double batch function
            auto batch_func = func_kernel->get_batch_func();
            debug_out() << "[DEBUG C++] matrix_from_gauss: Using double batch function, will call with n=" << (n * m) << " for entire grid" << std::endl;
            debug_out().flush();
            
            // Prepare all (x, y) pairs for 2D grid: xs[k] = gauss_x.x[i], ys[k] = gauss_y.x[j] where k = i*m + j
            std::vector<double> xs(n * m);
            std::vector<double> ys(n * m);
            std::vector<double> out(n * m);
            
            for (size_t i = 0; i < n; ++i) {
                double x_val = static_cast<double>(gauss_x.x[i]);
                for (size_t j = 0; j < m; ++j) {
                    size_t k = i * m + j;
                    xs[k] = x_val;
                    ys[k] = static_cast<double>(gauss_y.x[j]);
                }
            }
            
            // Evaluate entire grid in one batch call
            debug_out() << "[DEBUG C++] matrix_from_gauss: Calling batch_func with n=" << (n * m) << " for entire grid" << std::endl;
            debug_out().flush();
            batch_func(xs.data(), ys.data(), static_cast<int>(n * m), out.data(), user_data);
            debug_out() << "[DEBUG C++] matrix_from_gauss: batch_func returned" << std::endl;
            debug_out().flush();
            
            // Copy results to matrix
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    size_t k = i * m + j;
                    res(i, j) = static_cast<T>(out[k]);
                }
            }
            
            return res;
        } else if constexpr (std::is_same<T, xprec::DDouble>::value) {
            // Double-double precision: batch_func_dd is required (no fallback)
            auto batch_func_dd = func_kernel->get_batch_func_dd();
            if (!batch_func_dd) {
                throw std::runtime_error("batch_func_dd is required for DDouble precision but was not provided");
            }
            
            debug_out() << "[DEBUG C++] matrix_from_gauss: Using double-double batch function, will call with n=" << (n * m) << " for entire grid" << std::endl;
            debug_out().flush();
            
            // Prepare all (x, y) pairs for 2D grid: xs[k] = gauss_x.x[i], ys[k] = gauss_y.x[j] where k = i*m + j
            std::vector<double> xs_hi(n * m), xs_lo(n * m);
            std::vector<double> ys_hi(n * m), ys_lo(n * m);
            std::vector<double> out_hi(n * m), out_lo(n * m);
            
            for (size_t i = 0; i < n; ++i) {
                double x_hi = static_cast<double>(gauss_x.x[i].hi());
                double x_lo = static_cast<double>(gauss_x.x[i].lo());
                for (size_t j = 0; j < m; ++j) {
                    size_t k = i * m + j;
                    xs_hi[k] = x_hi;
                    xs_lo[k] = x_lo;
                    ys_hi[k] = static_cast<double>(gauss_y.x[j].hi());
                    ys_lo[k] = static_cast<double>(gauss_y.x[j].lo());
                }
            }
            
            // Evaluate entire grid in one batch call
            debug_out() << "[DEBUG C++] matrix_from_gauss: Calling batch_func_dd with n=" << (n * m) << " for entire grid" << std::endl;
            debug_out().flush();
            batch_func_dd(xs_hi.data(), xs_lo.data(),
                         ys_hi.data(), ys_lo.data(),
                         static_cast<int>(n * m),
                         out_hi.data(), out_lo.data(),
                         user_data);
            debug_out() << "[DEBUG C++] matrix_from_gauss: batch_func_dd returned" << std::endl;
            debug_out().flush();
            
            // Copy results to matrix
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < m; ++j) {
                    size_t k = i * m + j;
                    res(i, j) = xprec::DDouble(out_hi[k], out_lo[k]);
                }
            }
            
            return res;
        }
    }
    
    // Fallback: element-by-element computation for non-FunctionKernel (LogisticKernel, etc.)
    debug_out() << "[DEBUG C++] matrix_from_gauss: Using fallback element-by-element computation for non-FunctionKernel, n=" << n << ", m=" << m << std::endl;
    debug_out().flush();
    for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                res(i, j) =
                    kernel.compute(gauss_x.x[i], gauss_y.x[j],
                                   gauss_x.x_forward[i], gauss_x.x_backward[i]);
            }
    }

    return res;
}

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

// Function to provide SVE hints
template <typename T>
std::shared_ptr<AbstractSVEHints<T>>
sve_hints(const std::shared_ptr<const AbstractKernel> &kernel, double epsilon)
{
    // First check if the kernel itself is one of the supported types
    if (auto logistic =
            std::dynamic_pointer_cast<const LogisticKernel>(kernel)) {
        return std::make_shared<SVEHintsLogistic<T>>(*logistic, epsilon);
    }
    if (auto bose =
            std::dynamic_pointer_cast<const RegularizedBoseKernel>(kernel)) {
        return std::make_shared<SVEHintsRegularizedBose<T>>(*bose, epsilon);
    }

    // Then check derived kernels
    auto derived_kernels = kernel->get_derived_kernels();
    for (const auto &derived_kernel : derived_kernels) {
        if (auto logistic = std::dynamic_pointer_cast<const LogisticKernel>(
                derived_kernel)) {
            return std::make_shared<SVEHintsLogistic<T>>(*logistic, epsilon);
        }
        if (auto bose = std::dynamic_pointer_cast<const RegularizedBoseKernel>(
                derived_kernel)) {
            return std::make_shared<SVEHintsRegularizedBose<T>>(*bose, epsilon);
        }
    }

    // Special handling for ReducedKernel types
    if (auto reduced =
            std::dynamic_pointer_cast<const AbstractReducedKernelBase>(
                kernel)) {
        auto inner_kernel = reduced->get_inner_kernel();
        if (inner_kernel) {
            auto inner_hints = sve_hints<T>(inner_kernel, epsilon);
            return std::make_shared<SVEHintsReduced<T>>(inner_hints);
        }
    }

    // Special handling for FunctionKernel
    if (auto func_kernel =
            std::dynamic_pointer_cast<const FunctionKernel>(kernel)) {
        return func_kernel->template sve_hints<T>(epsilon);
    }

    throw std::runtime_error("Unsupported kernel type");
}

} // namespace sparseir

