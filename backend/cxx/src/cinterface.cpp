#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/utils.hpp"
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <algorithm>

inline void DEBUG_LOG(const std::string& msg) {
    const char* env = std::getenv("SPARSEIR_DEBUG");
    if (env && std::string(env) == "1") {
        std::cout << "[DEBUG] " << msg << std::endl;
    }
}

#include "cinterface_impl/helper_types.hpp"
#include "cinterface_impl/opaque_types.hpp"
#include "cinterface_impl/helper_funcs.hpp"

// Implementation of the C API
extern "C" {

spir_kernel* spir_logistic_kernel_new(double lambda, int* status)
{
    try {
        auto kernel_ptr = std::make_shared<sparseir::LogisticKernel>(lambda);
        std::shared_ptr<sparseir::AbstractKernel> abstract_kernel = _safe_static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);

        // Check if dynamic_cast works at this point
        auto check_logistic = _safe_dynamic_pointer_cast<sparseir::LogisticKernel>(abstract_kernel);

        *status = SPIR_COMPUTATION_SUCCESS;
        return create_kernel(abstract_kernel);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_logistic_kernel_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_logistic_kernel_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_kernel* spir_reg_bose_kernel_new(double lambda, int* status)
{
    DEBUG_LOG("Creating RegularizedBoseKernel with lambda=" + std::to_string(lambda));
    try {
        auto kernel_ptr = std::make_shared<sparseir::RegularizedBoseKernel>(lambda);
        auto abstract_kernel = _safe_static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_kernel(abstract_kernel);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_reg_bose_kernel_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_reg_bose_kernel_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int spir_kernel_get_domain(const spir_kernel *k, double *xmin, double *xmax,
                           double *ymin, double *ymax)
{
    DEBUG_LOG("spir_kernel_get_domain called with kernel=" + std::to_string(reinterpret_cast<uintptr_t>(k)));
    auto impl = get_impl_kernel(k);
    if (!impl) {
        DEBUG_LOG("Failed to get kernel implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        DEBUG_LOG("Getting xrange and yrange");
        auto xrange = impl->xrange();
        auto yrange = impl->yrange();

        DEBUG_LOG("Setting output values: xrange=(" +
                  std::to_string(xrange.first) + ", " + std::to_string(xrange.second) + "), yrange=(" +
                  std::to_string(yrange.first) + ", " + std::to_string(yrange.second) + ")");
        if (!xmin) {
            DEBUG_LOG("xmin is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        if (!xmax) {
            DEBUG_LOG("xmax is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        if (!ymin) {
            DEBUG_LOG("ymin is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        if (!ymax) {
            DEBUG_LOG("ymax is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        *xmin = xrange.first;
        *xmax = xrange.second;
        *ymin = yrange.first;
        *ymax = yrange.second;

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_get_domain: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_kernel_get_sve_hints_segments_x(const spir_kernel *k, double epsilon,
                                         double *segments, int *n_segments)
{
    try {
        if (!n_segments) {
            DEBUG_LOG("n_segments is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }

        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get kernel implementation");
            return SPIR_GET_IMPL_FAILED;
        }

        auto hints = sparseir::sve_hints<double>(impl, epsilon);
        auto segs = hints->segments_x();

        if (segments == nullptr) {
            // First call: return the number of segments
            *n_segments = static_cast<int>(segs.size());
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Second call: copy segments to output array
        if (*n_segments < static_cast<int>(segs.size())) {
            DEBUG_LOG("segments array is too small");
            return SPIR_INVALID_ARGUMENT;
        }

        for (size_t i = 0; i < segs.size(); ++i) {
            segments[i] = static_cast<double>(segs[i]);
        }
        *n_segments = static_cast<int>(segs.size());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_get_sve_hints_segments_x: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_get_sve_hints_segments_x");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_kernel_get_sve_hints_segments_y(const spir_kernel *k, double epsilon,
                                         double *segments, int *n_segments)
{
    try {
        if (!n_segments) {
            DEBUG_LOG("n_segments is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }

        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get kernel implementation");
            return SPIR_GET_IMPL_FAILED;
        }

        auto hints = sparseir::sve_hints<double>(impl, epsilon);
        auto segs = hints->segments_y();

        if (segments == nullptr) {
            // First call: return the number of segments
            *n_segments = static_cast<int>(segs.size());
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Second call: copy segments to output array
        if (*n_segments < static_cast<int>(segs.size())) {
            DEBUG_LOG("segments array is too small");
            return SPIR_INVALID_ARGUMENT;
        }

        for (size_t i = 0; i < segs.size(); ++i) {
            segments[i] = static_cast<double>(segs[i]);
        }
        *n_segments = static_cast<int>(segs.size());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_get_sve_hints_segments_y: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_get_sve_hints_segments_y");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_kernel_get_sve_hints_nsvals(const spir_kernel *k, double epsilon, int *nsvals)
{
    try {
        if (!nsvals) {
            DEBUG_LOG("nsvals is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }

        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get kernel implementation");
            return SPIR_GET_IMPL_FAILED;
        }

        auto hints = sparseir::sve_hints<double>(impl, epsilon);
        *nsvals = hints->nsvals();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_get_sve_hints_nsvals: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_get_sve_hints_nsvals");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_kernel_get_sve_hints_ngauss(const spir_kernel *k, double epsilon, int *ngauss)
{
    try {
        if (!ngauss) {
            DEBUG_LOG("ngauss is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }

        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get kernel implementation");
            return SPIR_GET_IMPL_FAILED;
        }

        auto hints = sparseir::sve_hints<double>(impl, epsilon);
        *ngauss = hints->ngauss();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_get_sve_hints_ngauss: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_get_sve_hints_ngauss");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_choose_working_type(double epsilon)
{
    try {
        if (std::isnan(epsilon) || epsilon < 1e-8) {
            return SPIR_TWORK_FLOAT64X2;
        } else {
            return SPIR_TWORK_FLOAT64;
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_choose_working_type: " + std::string(e.what()));
        return SPIR_TWORK_FLOAT64;  // Default to FLOAT64 on error
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_choose_working_type");
        return SPIR_TWORK_FLOAT64;  // Default to FLOAT64 on error
    }
}

spir_funcs* spir_funcs_from_piecewise_legendre(
    const double* segments, int n_segments,
    const double* coeffs, int nfuncs, int order,
    int* status)
{
    try {
        if (!segments || !coeffs || !status) {
            if (status) *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        if (n_segments < 1) {
            DEBUG_LOG("n_segments must be >= 1");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        if (nfuncs < 1) {
            DEBUG_LOG("nfuncs must be >= 1");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Create knots vector from segments
        Eigen::VectorXd knots(n_segments + 1);
        for (int i = 0; i <= n_segments; ++i) {
            knots(i) = segments[i];
        }

        // Verify segments are monotonically increasing
        for (int i = 1; i <= n_segments; ++i) {
            if (knots(i) <= knots(i-1)) {
                DEBUG_LOG("segments must be monotonically increasing");
                *status = SPIR_INVALID_ARGUMENT;
                return nullptr;
            }
        }

        // Create coefficient matrix: data is (nfuncs, n_segments)
        // Each column represents one segment's coefficients
        Eigen::MatrixXd data(nfuncs, n_segments);
        for (int seg = 0; seg < n_segments; ++seg) {
            for (int deg = 0; deg < nfuncs; ++deg) {
                // Layout: coeffs[seg * nfuncs + deg]
                data(deg, seg) = coeffs[seg * nfuncs + deg];
            }
        }

        // Create PiecewiseLegendrePoly (l=-1 means not specified)
        sparseir::PiecewiseLegendrePoly poly(data, knots, -1);

        // Create PiecewiseLegendrePolyVector (single function)
        std::vector<sparseir::PiecewiseLegendrePoly> polyvec = {poly};
        auto polyvec_ptr = std::make_shared<sparseir::PiecewiseLegendrePolyVector>(polyvec);

        // Wrap in PiecewiseLegendrePolyFunctions
        auto funcs_impl = std::make_shared<PiecewiseLegendrePolyFunctions>(polyvec_ptr);

        // Create spir_funcs
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_funcs(_safe_static_pointer_cast<AbstractContinuousFunctions>(funcs_impl));
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_from_piecewise_legendre: " + std::string(e.what()));
        if (status) *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_from_piecewise_legendre");
        if (status) *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int spir_gauss_legendre_rule_piecewise_double(
    int n,
    const double* segments, int n_segments,
    double* x, double* w,
    int* status)
{
    try {
        if (!segments || !x || !w || !status) {
            if (status) *status = SPIR_INVALID_ARGUMENT;
            return SPIR_INVALID_ARGUMENT;
        }

        if (n < 1) {
            DEBUG_LOG("n must be >= 1");
            *status = SPIR_INVALID_ARGUMENT;
            return SPIR_INVALID_ARGUMENT;
        }

        if (n_segments < 1) {
            DEBUG_LOG("n_segments must be >= 1");
            *status = SPIR_INVALID_ARGUMENT;
            return SPIR_INVALID_ARGUMENT;
        }

        // Convert segments to vector
        std::vector<double> segs_vec(segments, segments + n_segments + 1);

        // Verify segments are monotonically increasing
        for (int i = 1; i <= n_segments; ++i) {
            if (segs_vec[i] <= segs_vec[i-1]) {
                DEBUG_LOG("segments must be monotonically increasing");
                *status = SPIR_INVALID_ARGUMENT;
                return SPIR_INVALID_ARGUMENT;
            }
        }

        // Generate base rule with DDouble precision, then convert to double
        auto rule_dd = sparseir::legendre(n);
        auto rule = sparseir::convert_rule<double>(rule_dd);

        // Create piecewise rule
        auto piecewise_rule = rule.piecewise(segs_vec);

        // Copy to output arrays
        for (int i = 0; i < piecewise_rule.x.size(); ++i) {
            x[i] = piecewise_rule.x(i);
            w[i] = piecewise_rule.w(i);
        }

        *status = SPIR_COMPUTATION_SUCCESS;
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_gauss_legendre_rule_piecewise_double: " + std::string(e.what()));
        if (status) *status = SPIR_INTERNAL_ERROR;
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_gauss_legendre_rule_piecewise_double");
        if (status) *status = SPIR_INTERNAL_ERROR;
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_gauss_legendre_rule_piecewise_ddouble(
    int n,
    const double* segments, int n_segments,
    double* x_high, double* x_low,
    double* w_high, double* w_low,
    int* status)
{
    try {
        if (!segments || !x_high || !x_low || !w_high || !w_low || !status) {
            if (status) *status = SPIR_INVALID_ARGUMENT;
            return SPIR_INVALID_ARGUMENT;
        }

        if (n < 1) {
            DEBUG_LOG("n must be >= 1");
            *status = SPIR_INVALID_ARGUMENT;
            return SPIR_INVALID_ARGUMENT;
        }

        if (n_segments < 1) {
            DEBUG_LOG("n_segments must be >= 1");
            *status = SPIR_INVALID_ARGUMENT;
            return SPIR_INVALID_ARGUMENT;
        }

        // Convert segments to vector
        std::vector<double> segs_vec(segments, segments + n_segments + 1);

        // Verify segments are monotonically increasing
        for (int i = 1; i <= n_segments; ++i) {
            if (segs_vec[i] <= segs_vec[i-1]) {
                DEBUG_LOG("segments must be monotonically increasing");
                *status = SPIR_INVALID_ARGUMENT;
                return SPIR_INVALID_ARGUMENT;
            }
        }

        // Generate base rule with DDouble precision
        auto rule_dd = sparseir::legendre(n);

        // Convert segments to DDouble
        std::vector<xprec::DDouble> segs_dd(segs_vec.size());
        for (size_t i = 0; i < segs_vec.size(); ++i) {
            segs_dd[i] = xprec::DDouble(segs_vec[i]);
        }

        // Create piecewise rule
        auto piecewise_rule = rule_dd.piecewise(segs_dd);

        // Extract high and low parts
        for (int i = 0; i < piecewise_rule.x.size(); ++i) {
            x_high[i] = piecewise_rule.x(i).hi();
            x_low[i] = piecewise_rule.x(i).lo();
            w_high[i] = piecewise_rule.w(i).hi();
            w_low[i] = piecewise_rule.w(i).lo();
        }

        *status = SPIR_COMPUTATION_SUCCESS;
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_gauss_legendre_rule_piecewise_ddouble: " + std::string(e.what()));
        if (status) *status = SPIR_INTERNAL_ERROR;
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_gauss_legendre_rule_piecewise_ddouble");
        if (status) *status = SPIR_INTERNAL_ERROR;
        return SPIR_INTERNAL_ERROR;
    }
}

spir_sve_result* spir_sve_result_new(
    const spir_kernel *k,
    double epsilon,
    int lmax,
    int n_gauss,
    int Twork,
    int *status
)
{
    try {
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get a sve result implementation");
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // Set cutoff to NaN to use default value (2 * eps) internally
        double cutoff = std::numeric_limits<double>::quiet_NaN();

        if (lmax < 0) {
            lmax = std::numeric_limits<int>::max();
        }

        sparseir::TworkType Twork_type;
        if (Twork == SPIR_TWORK_FLOAT64) {
            Twork_type = sparseir::TworkType::FLOAT64;
        } else if (Twork == SPIR_TWORK_FLOAT64X2) {
            Twork_type = sparseir::TworkType::FLOAT64X2;
        } else if (Twork == SPIR_TWORK_AUTO) {
            Twork_type = sparseir::TworkType::AUTO;
        } else {
            DEBUG_LOG("Error: Invalid Twork");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        std::shared_ptr<sparseir::SVEResult> sve_result;

        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(
                *logistic, epsilon, cutoff, lmax, n_gauss, Twork_type
            ));
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(
                *bose, epsilon, cutoff, lmax, n_gauss, Twork_type
            ));
        } else {
            DEBUG_LOG("Unknown kernel type");
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        *status = SPIR_COMPUTATION_SUCCESS;
        return create_sve_result(sve_result);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_sve_result* spir_sve_result_from_matrix(
    const double* K_high, const double* K_low,
    int nx, int ny, int order,
    const double* segments_x, int n_segments_x,
    const double* segments_y, int n_segments_y,
    int n_gauss, double epsilon,
    int* status)
{
    try {
        // Input validation
        if (!K_high || !segments_x || !segments_y || !status) {
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        if (nx < 1 || ny < 1 || n_segments_x < 1 || n_segments_y < 1 || n_gauss < 1) {
            DEBUG_LOG("Invalid dimensions or parameters");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Verify segments are monotonically increasing
        for (int i = 1; i <= n_segments_x; ++i) {
            if (segments_x[i] <= segments_x[i-1]) {
                DEBUG_LOG("segments_x must be monotonically increasing");
                *status = SPIR_INVALID_ARGUMENT;
                return nullptr;
            }
        }
        for (int i = 1; i <= n_segments_y; ++i) {
            if (segments_y[i] <= segments_y[i-1]) {
                DEBUG_LOG("segments_y must be monotonically increasing");
                *status = SPIR_INVALID_ARGUMENT;
                return nullptr;
            }
        }

        // Determine if using DDouble precision
        bool use_ddouble = (K_low != nullptr);

        // Convert segments to vectors
        std::vector<double> segs_x_vec(segments_x, segments_x + n_segments_x + 1);
        std::vector<double> segs_y_vec(segments_y, segments_y + n_segments_y + 1);

        // Reconstruct Gauss rules
        auto rule_base_dd = sparseir::legendre(n_gauss);

        if (use_ddouble) {
            // DDouble precision
            using T = xprec::DDouble;

            // Convert segments to DDouble
            std::vector<T> segs_x_dd(segs_x_vec.size());
            std::vector<T> segs_y_dd(segs_y_vec.size());
            for (size_t i = 0; i < segs_x_vec.size(); ++i) {
                segs_x_dd[i] = T(segs_x_vec[i]);
            }
            for (size_t i = 0; i < segs_y_vec.size(); ++i) {
                segs_y_dd[i] = T(segs_y_vec[i]);
            }

            auto rule = sparseir::convert_rule<T>(rule_base_dd);
            auto gauss_x = rule.piecewise(segs_x_dd);
            auto gauss_y = rule.piecewise(segs_y_dd);

            // Convert input matrix K to Eigen::MatrixX<DDouble>
            Eigen::MatrixX<T> K(nx, ny);
            if (order == SPIR_ORDER_ROW_MAJOR) {
                for (int i = 0; i < nx; ++i) {
                    for (int j = 0; j < ny; ++j) {
                        int idx = i * ny + j;
                        K(i, j) = xprec::DDouble(K_high[idx], K_low[idx]);
                    }
                }
            } else { // COLUMN_MAJOR
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        int idx = j * nx + i;
                        K(i, j) = xprec::DDouble(K_high[idx], K_low[idx]);
                    }
                }
            }

            // Compute SVD
            auto svd_result = sparseir::compute_svd(K);
            auto u = std::get<0>(svd_result);
            auto s_ = std::get<1>(svd_result);
            auto v = std::get<2>(svd_result);

            // Postprocess similar to SamplingSVE::postprocess
            Eigen::VectorXd s = s_.template cast<double>();
            Eigen::VectorX<T> gauss_x_w = Eigen::VectorX<T>::Map(gauss_x.w.data(), gauss_x.w.size());
            Eigen::VectorX<T> gauss_y_w = Eigen::VectorX<T>::Map(gauss_y.w.data(), gauss_y.w.size());

            // Normalize u and v by weights
            Eigen::MatrixX<T> u_x_ = u;
            for (int i = 0; i < u_x_.rows(); ++i) {
                for (int j = 0; j < u_x_.cols(); ++j) {
                    u_x_(i, j) = u(i, j) / sparseir::sqrt_impl(gauss_x_w[i]);
                }
            }

            Eigen::MatrixX<T> v_y_ = v;
            for (int i = 0; i < v_y_.rows(); ++i) {
                for (int j = 0; j < v_y_.cols(); ++j) {
                    v_y_(i, j) = v(i, j) / sparseir::sqrt_impl(gauss_y_w[i]);
                }
            }

            // Reshape to Tensor
            Eigen::Tensor<T, 3> u_x(n_gauss, n_segments_x, s.size());
            Eigen::Tensor<T, 3> v_y(n_gauss, n_segments_y, s.size());

            for (int i = 0; i < u_x.dimension(0); ++i) {
                for (int j = 0; j < u_x.dimension(1); ++j) {
                    for (int k = 0; k < u_x.dimension(2); ++k) {
                        u_x(i, j, k) = u_x_(j * n_gauss + i, k);
                    }
                }
            }

            for (int i = 0; i < v_y.dimension(0); ++i) {
                for (int j = 0; j < v_y.dimension(1); ++j) {
                    for (int k = 0; k < v_y.dimension(2); ++k) {
                        v_y(i, j, k) = v_y_(j * n_gauss + i, k);
                    }
                }
            }

            // Apply Legendre collocation
            Eigen::MatrixX<T> cmat = sparseir::legendre_collocation<T>(rule);
            Eigen::Tensor<T, 3> u_data(cmat.rows(), n_segments_x, s.size());
            Eigen::Tensor<T, 3> v_data(cmat.rows(), n_segments_y, s.size());

            for (int j = 0; j < u_data.dimension(1); ++j) {
                for (int k = 0; k < u_data.dimension(2); ++k) {
                    for (int i = 0; i < u_data.dimension(0); ++i) {
                        u_data(i, j, k) = T(0);
                        for (int l = 0; l < cmat.cols(); ++l) {
                            u_data(i, j, k) += cmat(i, l) * u_x(l, j, k);
                        }
                    }
                }
            }

            for (int j = 0; j < v_data.dimension(1); ++j) {
                for (int k = 0; k < v_data.dimension(2); ++k) {
                    for (int i = 0; i < v_data.dimension(0); ++i) {
                        v_data(i, j, k) = T(0);
                        for (int l = 0; l < cmat.cols(); ++l) {
                            v_data(i, j, k) += cmat(i, l) * v_y(l, j, k);
                        }
                    }
                }
            }

            // Apply segment scaling
            auto dsegs_x = sparseir::diff(segs_x_dd);
            auto dsegs_y = sparseir::diff(segs_y_dd);

            for (int j = 0; j < u_data.dimension(1); ++j) {
                for (int i = 0; i < u_data.dimension(0); ++i) {
                    for (int k = 0; k < u_data.dimension(2); ++k) {
                        u_data(i, j, k) *= sparseir::sqrt_impl(T(0.5) * dsegs_x[j]);
                    }
                }
            }

            for (int j = 0; j < v_data.dimension(1); ++j) {
                for (int i = 0; i < v_data.dimension(0); ++i) {
                    for (int k = 0; k < v_data.dimension(2); ++k) {
                        v_data(i, j, k) *= sparseir::sqrt_impl(T(0.5) * dsegs_y[j]);
                    }
                }
            }

            // Convert to PiecewiseLegendrePoly
            std::vector<sparseir::PiecewiseLegendrePoly> polyvec_u;
            std::vector<sparseir::PiecewiseLegendrePoly> polyvec_v;
            Eigen::VectorXd knots_x = Eigen::Map<Eigen::VectorXd>(segs_x_vec.data(), segs_x_vec.size());
            Eigen::VectorXd knots_y = Eigen::Map<Eigen::VectorXd>(segs_y_vec.data(), segs_y_vec.size());

            for (int i = 0; i < u_data.dimension(2); ++i) {
                Eigen::MatrixXd slice_double(u_data.dimension(0), u_data.dimension(1));
                for (int j = 0; j < u_data.dimension(0); ++j) {
                    for (int k = 0; k < u_data.dimension(1); ++k) {
                        slice_double(j, k) = static_cast<double>(u_data(j, k, i));
                    }
                }
                polyvec_u.push_back(
                    sparseir::PiecewiseLegendrePoly(slice_double, knots_x, i, sparseir::diff(knots_x)));
            }

            for (int i = 0; i < v_data.dimension(2); ++i) {
                Eigen::MatrixXd slice_double(v_data.dimension(0), v_data.dimension(1));
                for (int j = 0; j < v_data.dimension(0); ++j) {
                    for (int k = 0; k < v_data.dimension(1); ++k) {
                        slice_double(j, k) = static_cast<double>(v_data(j, k, i));
                    }
                }
                polyvec_v.push_back(
                    sparseir::PiecewiseLegendrePoly(slice_double, knots_y, i, sparseir::diff(knots_y)));
            }

            sparseir::PiecewiseLegendrePolyVector ulx(polyvec_u);
            sparseir::PiecewiseLegendrePolyVector vly(polyvec_v);
            sparseir::canonicalize(ulx, vly);

            auto sve_result = std::make_shared<sparseir::SVEResult>(ulx, s, vly, epsilon);
            *status = SPIR_COMPUTATION_SUCCESS;
            return create_sve_result(sve_result);

        } else {
            // Double precision
            using T = double;

            auto rule = sparseir::convert_rule<T>(rule_base_dd);
            auto gauss_x = rule.piecewise(segs_x_vec);
            auto gauss_y = rule.piecewise(segs_y_vec);

            // Convert input matrix K to Eigen::MatrixXd
            Eigen::MatrixXd K(nx, ny);
            if (order == SPIR_ORDER_ROW_MAJOR) {
                for (int i = 0; i < nx; ++i) {
                    for (int j = 0; j < ny; ++j) {
                        K(i, j) = K_high[i * ny + j];
                    }
                }
            } else { // COLUMN_MAJOR
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        K(i, j) = K_high[j * nx + i];
                    }
                }
            }

            // Compute SVD
            auto svd_result = sparseir::compute_svd(K);
            auto u = std::get<0>(svd_result);
            auto s_ = std::get<1>(svd_result);
            auto v = std::get<2>(svd_result);

            // Postprocess similar to SamplingSVE::postprocess
            Eigen::VectorXd s = s_;
            Eigen::VectorXd gauss_x_w = Eigen::Map<Eigen::VectorXd>(gauss_x.w.data(), gauss_x.w.size());
            Eigen::VectorXd gauss_y_w = Eigen::Map<Eigen::VectorXd>(gauss_y.w.data(), gauss_y.w.size());

            // Normalize u and v by weights
            Eigen::MatrixXd u_x_ = u;
            for (int i = 0; i < u_x_.rows(); ++i) {
                for (int j = 0; j < u_x_.cols(); ++j) {
                    u_x_(i, j) = u(i, j) / std::sqrt(gauss_x_w[i]);
                }
            }

            Eigen::MatrixXd v_y_ = v;
            for (int i = 0; i < v_y_.rows(); ++i) {
                for (int j = 0; j < v_y_.cols(); ++j) {
                    v_y_(i, j) = v(i, j) / std::sqrt(gauss_y_w[i]);
                }
            }

            // Reshape to Tensor
            Eigen::Tensor<double, 3> u_x(n_gauss, n_segments_x, s.size());
            Eigen::Tensor<double, 3> v_y(n_gauss, n_segments_y, s.size());

            for (int i = 0; i < u_x.dimension(0); ++i) {
                for (int j = 0; j < u_x.dimension(1); ++j) {
                    for (int k = 0; k < u_x.dimension(2); ++k) {
                        u_x(i, j, k) = u_x_(j * n_gauss + i, k);
                    }
                }
            }

            for (int i = 0; i < v_y.dimension(0); ++i) {
                for (int j = 0; j < v_y.dimension(1); ++j) {
                    for (int k = 0; k < v_y.dimension(2); ++k) {
                        v_y(i, j, k) = v_y_(j * n_gauss + i, k);
                    }
                }
            }

            // Apply Legendre collocation
            Eigen::MatrixXd cmat = sparseir::legendre_collocation<double>(rule);
            Eigen::Tensor<double, 3> u_data(cmat.rows(), n_segments_x, s.size());
            Eigen::Tensor<double, 3> v_data(cmat.rows(), n_segments_y, s.size());

            for (int j = 0; j < u_data.dimension(1); ++j) {
                for (int k = 0; k < u_data.dimension(2); ++k) {
                    for (int i = 0; i < u_data.dimension(0); ++i) {
                        u_data(i, j, k) = 0.0;
                        for (int l = 0; l < cmat.cols(); ++l) {
                            u_data(i, j, k) += cmat(i, l) * u_x(l, j, k);
                        }
                    }
                }
            }

            for (int j = 0; j < v_data.dimension(1); ++j) {
                for (int k = 0; k < v_data.dimension(2); ++k) {
                    for (int i = 0; i < v_data.dimension(0); ++i) {
                        v_data(i, j, k) = 0.0;
                        for (int l = 0; l < cmat.cols(); ++l) {
                            v_data(i, j, k) += cmat(i, l) * v_y(l, j, k);
                        }
                    }
                }
            }

            // Apply segment scaling
            std::vector<double> dsegs_x(n_segments_x);
            std::vector<double> dsegs_y(n_segments_y);
            for (int i = 0; i < n_segments_x; ++i) {
                dsegs_x[i] = segs_x_vec[i+1] - segs_x_vec[i];
            }
            for (int i = 0; i < n_segments_y; ++i) {
                dsegs_y[i] = segs_y_vec[i+1] - segs_y_vec[i];
            }

            for (int j = 0; j < u_data.dimension(1); ++j) {
                for (int i = 0; i < u_data.dimension(0); ++i) {
                    for (int k = 0; k < u_data.dimension(2); ++k) {
                        u_data(i, j, k) *= std::sqrt(0.5 * dsegs_x[j]);
                    }
                }
            }

            for (int j = 0; j < v_data.dimension(1); ++j) {
                for (int i = 0; i < v_data.dimension(0); ++i) {
                    for (int k = 0; k < v_data.dimension(2); ++k) {
                        v_data(i, j, k) *= std::sqrt(0.5 * dsegs_y[j]);
                    }
                }
            }

            // Convert to PiecewiseLegendrePoly
            std::vector<sparseir::PiecewiseLegendrePoly> polyvec_u;
            std::vector<sparseir::PiecewiseLegendrePoly> polyvec_v;
            Eigen::VectorXd knots_x = Eigen::Map<Eigen::VectorXd>(segs_x_vec.data(), segs_x_vec.size());
            Eigen::VectorXd knots_y = Eigen::Map<Eigen::VectorXd>(segs_y_vec.data(), segs_y_vec.size());

            for (int i = 0; i < u_data.dimension(2); ++i) {
                Eigen::MatrixXd slice_double(u_data.dimension(0), u_data.dimension(1));
                for (int j = 0; j < u_data.dimension(0); ++j) {
                    for (int k = 0; k < u_data.dimension(1); ++k) {
                        slice_double(j, k) = u_data(j, k, i);
                    }
                }
                polyvec_u.push_back(
                    sparseir::PiecewiseLegendrePoly(slice_double, knots_x, i, sparseir::diff(knots_x)));
            }

            for (int i = 0; i < v_data.dimension(2); ++i) {
                Eigen::MatrixXd slice_double(v_data.dimension(0), v_data.dimension(1));
                for (int j = 0; j < v_data.dimension(0); ++j) {
                    for (int k = 0; k < v_data.dimension(1); ++k) {
                        slice_double(j, k) = v_data(j, k, i);
                    }
                }
                polyvec_v.push_back(
                    sparseir::PiecewiseLegendrePoly(slice_double, knots_y, i, sparseir::diff(knots_y)));
            }

            sparseir::PiecewiseLegendrePolyVector ulx(polyvec_u);
            sparseir::PiecewiseLegendrePolyVector vly(polyvec_v);
            sparseir::canonicalize(ulx, vly);

            auto sve_result = std::make_shared<sparseir::SVEResult>(ulx, s, vly, epsilon);
            *status = SPIR_COMPUTATION_SUCCESS;
            return create_sve_result(sve_result);
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_from_matrix: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_from_matrix");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_sve_result* spir_sve_result_from_matrix_centrosymmetric(
    const double* K_even_high, const double* K_even_low,
    const double* K_odd_high, const double* K_odd_low,
    int nx, int ny, int order,
    const double* segments_x, int n_segments_x,
    const double* segments_y, int n_segments_y,
    int n_gauss, double epsilon,
    int* status)
{
    try {
        // Input validation
        if (!K_even_high || !K_odd_high || !segments_x || !segments_y || !status) {
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Validate segments for centrosymmetric kernels:
        // segments_x and segments_y must start at 0 and end at positive values
        // This is required because even/odd matrices are computed on [0, xmax] x [0, ymax]
        const double eps_tol = 1e-12;

        // Check segments_x
        if (n_segments_x < 1) {
            DEBUG_LOG("Invalid n_segments_x: must be at least 1, got " + std::to_string(n_segments_x));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }
        if (std::abs(segments_x[0]) > eps_tol) {
            DEBUG_LOG("segments_x must start at 0, but got " + std::to_string(segments_x[0]));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }
        if (segments_x[n_segments_x] <= 0.0) {
            DEBUG_LOG("segments_x must end at a positive value, but got " + std::to_string(segments_x[n_segments_x]));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Check segments_y
        if (n_segments_y < 1) {
            DEBUG_LOG("Invalid n_segments_y: must be at least 1, got " + std::to_string(n_segments_y));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }
        if (std::abs(segments_y[0]) > eps_tol) {
            DEBUG_LOG("segments_y must start at 0, but got " + std::to_string(segments_y[0]));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }
        if (segments_y[n_segments_y] <= 0.0) {
            DEBUG_LOG("segments_y must end at a positive value, but got " + std::to_string(segments_y[n_segments_y]));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Verify segments are monotonically increasing (additional check)
        for (int i = 1; i <= n_segments_x; ++i) {
            if (segments_x[i] <= segments_x[i-1]) {
                DEBUG_LOG("segments_x must be monotonically increasing, but segments_x[" +
                         std::to_string(i-1) + "]=" + std::to_string(segments_x[i-1]) +
                         " >= segments_x[" + std::to_string(i) + "]=" + std::to_string(segments_x[i]));
                *status = SPIR_INVALID_ARGUMENT;
                return nullptr;
            }
        }
        for (int i = 1; i <= n_segments_y; ++i) {
            if (segments_y[i] <= segments_y[i-1]) {
                DEBUG_LOG("segments_y must be monotonically increasing, but segments_y[" +
                         std::to_string(i-1) + "]=" + std::to_string(segments_y[i-1]) +
                         " >= segments_y[" + std::to_string(i) + "]=" + std::to_string(segments_y[i]));
                *status = SPIR_INVALID_ARGUMENT;
                return nullptr;
            }
        }

        // Debug: Check matrix sizes and non-zero elements
        DEBUG_LOG("Input matrices: nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny));
        DEBUG_LOG("n_segments_x=" + std::to_string(n_segments_x) + ", n_segments_y=" + std::to_string(n_segments_y));
        DEBUG_LOG("n_gauss=" + std::to_string(n_gauss));
        if (segments_x && n_segments_x > 0) {
            DEBUG_LOG("segments_x[0]=" + std::to_string(segments_x[0]) +
                     ", segments_x[" + std::to_string(n_segments_x) + "]=" +
                     std::to_string(segments_x[n_segments_x]));
        }
        if (segments_y && n_segments_y > 0) {
            DEBUG_LOG("segments_y[0]=" + std::to_string(segments_y[0]) +
                     ", segments_y[" + std::to_string(n_segments_y) + "]=" +
                     std::to_string(segments_y[n_segments_y]));
        }

        // Check if matrices are non-empty
        if (nx <= 0 || ny <= 0) {
            DEBUG_LOG("Error: Empty matrices: nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Compute SVE for even and odd matrices separately
        int status_even, status_odd;
        DEBUG_LOG("Computing SVE for even matrix...");
        auto sve_even = spir_sve_result_from_matrix(
            K_even_high, K_even_low, nx, ny, order,
            segments_x, n_segments_x, segments_y, n_segments_y,
            n_gauss, epsilon, &status_even);

        if (status_even != SPIR_COMPUTATION_SUCCESS || !sve_even) {
            DEBUG_LOG("Error: Failed to compute SVE for even matrix: status=" + std::to_string(status_even));
            *status = status_even;
            return nullptr;
        }
        DEBUG_LOG("Successfully computed SVE for even matrix");

        DEBUG_LOG("Computing SVE for odd matrix...");
        auto sve_odd = spir_sve_result_from_matrix(
            K_odd_high, K_odd_low, nx, ny, order,
            segments_x, n_segments_x, segments_y, n_segments_y,
            n_gauss, epsilon, &status_odd);

        if (status_odd != SPIR_COMPUTATION_SUCCESS || !sve_odd) {
            DEBUG_LOG("Error: Failed to compute SVE for odd matrix: status=" + std::to_string(status_odd));
            spir_sve_result_release(sve_even);
            *status = status_odd;
            return nullptr;
        }
        DEBUG_LOG("Successfully computed SVE for odd matrix");

        // Get implementations
        auto sve_even_impl = get_impl_sve_result(sve_even);
        auto sve_odd_impl = get_impl_sve_result(sve_odd);

        if (!sve_even_impl || !sve_odd_impl) {
            DEBUG_LOG("Error: Failed to get SVE result implementations");
            spir_sve_result_release(sve_even);
            spir_sve_result_release(sve_odd);
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // Debug: Check sizes of even and odd results
        DEBUG_LOG("Even SVE result: s.size()=" + std::to_string(sve_even_impl->s.size()) +
                 ", u->size()=" + std::to_string(sve_even_impl->u->size()) +
                 ", v->size()=" + std::to_string(sve_even_impl->v->size()));
        DEBUG_LOG("Odd SVE result: s.size()=" + std::to_string(sve_odd_impl->s.size()) +
                 ", u->size()=" + std::to_string(sve_odd_impl->u->size()) +
                 ", v->size()=" + std::to_string(sve_odd_impl->v->size()));

        // Check if even and odd results are non-empty
        if (sve_even_impl->s.size() == 0 || sve_even_impl->u->size() == 0 || sve_even_impl->v->size() == 0) {
            DEBUG_LOG("Error: Even SVE result is empty");
            spir_sve_result_release(sve_even);
            spir_sve_result_release(sve_odd);
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }
        if (sve_odd_impl->s.size() == 0 || sve_odd_impl->u->size() == 0 || sve_odd_impl->v->size() == 0) {
            DEBUG_LOG("Error: Odd SVE result is empty");
            spir_sve_result_release(sve_even);
            spir_sve_result_release(sve_odd);
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        // Merge even and odd results similar to CentrosymmSVE::postprocess
        std::vector<sparseir::PiecewiseLegendrePoly> u_merged;
        u_merged.reserve(sve_even_impl->u->size() + sve_odd_impl->u->size());
        u_merged.insert(u_merged.end(), sve_even_impl->u->begin(), sve_even_impl->u->end());
        u_merged.insert(u_merged.end(), sve_odd_impl->u->begin(), sve_odd_impl->u->end());

        Eigen::VectorXd s_merged(sve_even_impl->s.size() + sve_odd_impl->s.size());
        s_merged << sve_even_impl->s, sve_odd_impl->s;

        std::vector<sparseir::PiecewiseLegendrePoly> v_merged;
        v_merged.reserve(sve_even_impl->v->size() + sve_odd_impl->v->size());
        v_merged.insert(v_merged.end(), sve_even_impl->v->begin(), sve_even_impl->v->end());
        v_merged.insert(v_merged.end(), sve_odd_impl->v->begin(), sve_odd_impl->v->end());

        sparseir::PiecewiseLegendrePolyVector _u_complete(u_merged);
        sparseir::PiecewiseLegendrePolyVector _v_complete(v_merged);

        Eigen::VectorXi sign_even = Eigen::VectorXi::Ones(sve_even_impl->s.size());
        Eigen::VectorXi sign_odd = -Eigen::VectorXi::Ones(sve_odd_impl->s.size());
        Eigen::VectorXi signs = Eigen::VectorXi::Zero(s_merged.size());
        signs << sign_even, sign_odd;

        // Sort by singular values (descending)
        std::vector<size_t> sorted_indices = sparseir::sortperm_rev(s_merged);

        std::vector<sparseir::PiecewiseLegendrePoly> u_sorted(sorted_indices.size());
        std::vector<sparseir::PiecewiseLegendrePoly> v_sorted(sorted_indices.size());
        Eigen::VectorXi signs_sorted(sorted_indices.size());
        Eigen::VectorXd s_sorted(sorted_indices.size());

        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            u_sorted[i] = u_merged[sorted_indices[i]];
            v_sorted[i] = v_merged[sorted_indices[i]];
            s_sorted[i] = s_merged[sorted_indices[i]];
            signs_sorted[i] = signs[sorted_indices[i]];
        }

        // Convert segments to vectors
        std::vector<double> segs_x_vec(segments_x, segments_x + n_segments_x + 1);
        std::vector<double> segs_y_vec(segments_y, segments_y + n_segments_y + 1);
        Eigen::VectorXd segs_x = Eigen::Map<Eigen::VectorXd>(segs_x_vec.data(), segs_x_vec.size());
        Eigen::VectorXd segs_y = Eigen::Map<Eigen::VectorXd>(segs_y_vec.data(), segs_y_vec.size());

        // Build complete domain polynomials
        std::vector<sparseir::PiecewiseLegendrePoly> u_complete_vec;
        std::vector<sparseir::PiecewiseLegendrePoly> v_complete_vec;

        // Check if we have any singular values
        if (u_sorted.empty()) {
            spir_sve_result_release(sve_even);
            spir_sve_result_release(sve_odd);
            *status = SPIR_INTERNAL_ERROR;
            DEBUG_LOG("No singular values found in merged SVE result");
            return nullptr;
        }

        Eigen::VectorXd poly_flip_x(u_sorted[0].data.rows());
        for (int i = 0; i < u_sorted[0].data.rows(); ++i) {
            poly_flip_x(i) = (i % 2 == 0) ? 1.0 : -1.0;
        }

        DEBUG_LOG("Processing " + std::to_string(u_sorted.size()) + " sorted singular values");

        for (size_t i = 0; i < u_sorted.size(); ++i) {
            try {
                DEBUG_LOG("Processing singular value " + std::to_string(i) + "/" + std::to_string(u_sorted.size()));
            Eigen::MatrixXd u_pos_data = u_sorted[i].data / std::sqrt(2);
            Eigen::MatrixXd v_pos_data = v_sorted[i].data / std::sqrt(2);

                DEBUG_LOG("u_pos_data size: " + std::to_string(u_pos_data.rows()) + "x" + std::to_string(u_pos_data.cols()));
                DEBUG_LOG("v_pos_data size: " + std::to_string(v_pos_data.rows()) + "x" + std::to_string(v_pos_data.cols()));

                // Check dimensions
                if (u_pos_data.cols() == 0 || v_pos_data.cols() == 0) {
                    DEBUG_LOG("Zero columns in u_pos_data or v_pos_data at i=" + std::to_string(i));
                    spir_sve_result_release(sve_even);
                    spir_sve_result_release(sve_odd);
                    *status = SPIR_INTERNAL_ERROR;
                    return nullptr;
                }

            Eigen::MatrixXd u_neg_data = u_pos_data.rowwise().reverse();
            u_neg_data = u_neg_data.array().colwise() * (poly_flip_x * signs_sorted[i]).array();
            Eigen::MatrixXd v_neg_data = v_pos_data.rowwise().reverse();
            v_neg_data = v_neg_data.array().colwise() * (poly_flip_x * signs_sorted[i]).array();

                DEBUG_LOG("u_neg_data size: " + std::to_string(u_neg_data.rows()) + "x" + std::to_string(u_neg_data.cols()));
                DEBUG_LOG("v_neg_data size: " + std::to_string(v_neg_data.rows()) + "x" + std::to_string(v_neg_data.cols()));

                // The merged data should have the same number of columns as the full domain segments
                // u_pos_data already covers the full domain, so we shouldn't double the columns
                // Instead, we need to properly extend to negative side
                // Check if segments are symmetric
                DEBUG_LOG("segs_x.size()=" + std::to_string(segs_x.size()) + ", u_pos_data.cols()=" + std::to_string(u_pos_data.cols()));
                if (segs_x.size() != u_pos_data.cols() + 1) {
                    DEBUG_LOG("Segment size mismatch: segs_x.size()=" + std::to_string(segs_x.size()) +
                             ", u_pos_data.cols()=" + std::to_string(u_pos_data.cols()));
                    // Try to construct symmetric segments from positive half
                    // For now, use the existing data structure
            Eigen::MatrixXd u_data = Eigen::MatrixXd::Zero(
                u_pos_data.rows(), u_neg_data.cols() + u_pos_data.cols());
            u_data.leftCols(u_neg_data.cols()) = u_neg_data;
            u_data.rightCols(u_pos_data.cols()) = u_pos_data;
            Eigen::MatrixXd v_data = Eigen::MatrixXd::Zero(
                v_pos_data.rows(), v_neg_data.cols() + v_pos_data.cols());
            v_data.leftCols(v_neg_data.cols()) = v_neg_data;
            v_data.rightCols(v_pos_data.cols()) = v_pos_data;

                    // Construct symmetric segments
                    Eigen::VectorXd segs_x_symmetric = Eigen::VectorXd::Zero(u_data.cols() + 1);
                    int half = u_pos_data.cols();
                    segs_x_symmetric.head(half) = -segs_x.tail(half).reverse();
                    segs_x_symmetric.tail(half + 1) = segs_x.tail(half + 1);

                    Eigen::VectorXd segs_y_symmetric = Eigen::VectorXd::Zero(v_data.cols() + 1);
                    half = v_pos_data.cols();
                    segs_y_symmetric.head(half) = -segs_y.tail(half).reverse();
                    segs_y_symmetric.tail(half + 1) = segs_y.tail(half + 1);

                    Eigen::VectorXd segs_x_diff = segs_x_symmetric.tail(segs_x_symmetric.size() - 1) - segs_x_symmetric.head(segs_x_symmetric.size() - 1);
                    Eigen::VectorXd segs_y_diff = segs_y_symmetric.tail(segs_y_symmetric.size() - 1) - segs_y_symmetric.head(segs_y_symmetric.size() - 1);

            u_complete_vec.push_back(
                        sparseir::PiecewiseLegendrePoly(u_data, segs_x_symmetric, static_cast<int>(i), segs_x_diff, signs_sorted[i]));
            v_complete_vec.push_back(
                        sparseir::PiecewiseLegendrePoly(v_data, segs_y_symmetric, static_cast<int>(i), segs_y_diff, signs_sorted[i]));
                } else {
                    // Segments already match, but we need to extend to full domain [-xmax, xmax]
                    // Current segments are [0, xmax], need to construct full domain
                    DEBUG_LOG("Segments match, extending to full domain");
                    DEBUG_LOG("Current segs_x: [" + std::to_string(segs_x(0)) + ", ..., " + std::to_string(segs_x(segs_x.size()-1)) + "]");

                    // Construct symmetric segments for full domain
                    int half = u_pos_data.cols();
                    Eigen::VectorXd segs_x_full = Eigen::VectorXd::Zero(2 * half + 1);
                    segs_x_full.head(half) = -segs_x.tail(half).reverse();
                    segs_x_full(half) = 0.0;
                    segs_x_full.tail(half) = segs_x.tail(half);

                    int half_y = v_pos_data.cols();
                    Eigen::VectorXd segs_y_full = Eigen::VectorXd::Zero(2 * half_y + 1);
                    segs_y_full.head(half_y) = -segs_y.tail(half_y).reverse();
                    segs_y_full(half_y) = 0.0;
                    segs_y_full.tail(half_y) = segs_y.tail(half_y);

                    // Combine u_neg and u_pos data
                    Eigen::MatrixXd u_data = Eigen::MatrixXd::Zero(
                        u_pos_data.rows(), u_neg_data.cols() + u_pos_data.cols());
                    u_data.leftCols(u_neg_data.cols()) = u_neg_data;
                    u_data.rightCols(u_pos_data.cols()) = u_pos_data;
                    Eigen::MatrixXd v_data = Eigen::MatrixXd::Zero(
                        v_pos_data.rows(), v_neg_data.cols() + v_pos_data.cols());
                    v_data.leftCols(v_neg_data.cols()) = v_neg_data;
                    v_data.rightCols(v_pos_data.cols()) = v_pos_data;

                    DEBUG_LOG("u_data size: " + std::to_string(u_data.rows()) + "x" + std::to_string(u_data.cols()) +
                             ", segs_x_full size: " + std::to_string(segs_x_full.size()));
                    DEBUG_LOG("v_data size: " + std::to_string(v_data.rows()) + "x" + std::to_string(v_data.cols()) +
                             ", segs_y_full size: " + std::to_string(segs_y_full.size()));

                    Eigen::VectorXd segs_x_diff = segs_x_full.tail(segs_x_full.size() - 1) - segs_x_full.head(segs_x_full.size() - 1);
                    Eigen::VectorXd segs_y_diff = segs_y_full.tail(segs_y_full.size() - 1) - segs_y_full.head(segs_y_full.size() - 1);

                    DEBUG_LOG("Creating PiecewiseLegendrePoly...");
                    u_complete_vec.push_back(
                        sparseir::PiecewiseLegendrePoly(u_data, segs_x_full, static_cast<int>(i), segs_x_diff, signs_sorted[i]));
                    v_complete_vec.push_back(
                        sparseir::PiecewiseLegendrePoly(v_data, segs_y_full, static_cast<int>(i), segs_y_diff, signs_sorted[i]));
                    DEBUG_LOG("Successfully created PiecewiseLegendrePoly for singular value " + std::to_string(i));
                }
                DEBUG_LOG("Successfully processed singular value " + std::to_string(i));
            } catch (const std::exception &e) {
                DEBUG_LOG("Exception while processing singular value " + std::to_string(i) + ": " + std::string(e.what()));
                std::cerr << "[ERROR] Exception while processing singular value " << i << ": " << e.what() << std::endl;
                spir_sve_result_release(sve_even);
                spir_sve_result_release(sve_odd);
                *status = SPIR_INTERNAL_ERROR;
                return nullptr;
            }
        }

        sparseir::PiecewiseLegendrePolyVector u_complete(u_complete_vec);
        sparseir::PiecewiseLegendrePolyVector v_complete(v_complete_vec);

        auto sve_result = std::make_shared<sparseir::SVEResult>(u_complete, s_sorted, v_complete, epsilon);

        // Clean up temporary results
        spir_sve_result_release(sve_even);
        spir_sve_result_release(sve_odd);

        *status = SPIR_COMPUTATION_SUCCESS;
        return create_sve_result(sve_result);
    } catch (const std::exception &e) {
        std::string error_msg = "Exception in spir_sve_result_from_matrix_centrosymmetric: " + std::string(e.what());
        DEBUG_LOG(error_msg);
        std::cerr << "[ERROR] " << error_msg << std::endl;
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (const std::string &e) {
        std::string error_msg = "String exception in spir_sve_result_from_matrix_centrosymmetric: " + e;
        DEBUG_LOG(error_msg);
        std::cerr << "[ERROR] " << error_msg << std::endl;
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        std::string error_msg = "Unknown exception in spir_sve_result_from_matrix_centrosymmetric";
        DEBUG_LOG(error_msg);
        std::cerr << "[ERROR] " << error_msg << std::endl;
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_basis* spir_basis_new_from_sve_and_inv_weight(
    int statistics, double beta, double omega_max, double epsilon,
    double lambda, int ypower, double conv_radius,
    const spir_sve_result *sve,
    const spir_funcs *inv_weight_funcs,
    int max_size,
    int *status)
{
    DEBUG_LOG("spir_basis_new_from_sve_and_inv_weight called");
    DEBUG_LOG("  statistics=" + std::to_string(statistics) +
              ", beta=" + std::to_string(beta) +
              ", omega_max=" + std::to_string(omega_max) +
              ", epsilon=" + std::to_string(epsilon) +
              ", lambda=" + std::to_string(lambda) +
              ", ypower=" + std::to_string(ypower) +
              ", conv_radius=" + std::to_string(conv_radius) +
              ", max_size=" + std::to_string(max_size));
    DEBUG_LOG("  sve=" + (sve ? std::to_string(reinterpret_cast<uintptr_t>(sve)) : std::string("null")));
    DEBUG_LOG("  inv_weight_funcs=" + (inv_weight_funcs ? std::to_string(reinterpret_cast<uintptr_t>(inv_weight_funcs)) : std::string("null")));
    DEBUG_LOG("  status=" + (status ? std::to_string(reinterpret_cast<uintptr_t>(status)) : std::string("null")));

    try {
        // Input validation
        if (!sve || !inv_weight_funcs || !status) {
            DEBUG_LOG("Input validation failed: null pointer");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        if (beta <= 0.0 || omega_max < 0.0) {
            DEBUG_LOG("Invalid beta or omega_max: beta=" + std::to_string(beta) + ", omega_max=" + std::to_string(omega_max));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Get SVE result implementation
        DEBUG_LOG("Getting SVE result implementation");
        auto sve_impl = get_impl_sve_result(sve);
        if (!sve_impl) {
            DEBUG_LOG("Failed to get SVE result implementation");
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }
        DEBUG_LOG("SVE result implementation obtained successfully");

        // Create inv_weight_func from spir_funcs
        // inv_weight_funcs represents inv_weight_func(omega) for fixed beta
        // Create omega-only function that evaluates spir_funcs
        DEBUG_LOG("Creating inv_weight_func from spir_funcs");
        std::function<double(double)> inv_weight_func =
            [inv_weight_funcs](double omega) -> double {
                int funcs_size;
                int eval_status = spir_funcs_get_size(inv_weight_funcs, &funcs_size);
                if (eval_status != SPIR_COMPUTATION_SUCCESS || funcs_size < 1) {
                    return 1.0; // Default to 1.0 on error
                }

                // Evaluate at omega (treating omega as x coordinate)
                std::vector<double> values(funcs_size);
                eval_status = spir_funcs_eval(inv_weight_funcs, omega, values.data());
                if (eval_status != SPIR_COMPUTATION_SUCCESS) {
                    return 1.0; // Default to 1.0 on error
                }

                // Return the first function value (assuming single function)
                return values[0];
            };

        // Create basis using new constructor
        DEBUG_LOG("Creating FiniteTempBasis with statistics=" + std::to_string(statistics));
        spir_basis* result = nullptr;
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            using FiniteTempBasisType = sparseir::FiniteTempBasis<sparseir::Fermionic>;
            auto impl = std::make_shared<FiniteTempBasisType>(
                beta, omega_max, epsilon, lambda, ypower, conv_radius,
                *sve_impl, inv_weight_func, max_size);
            result = create_basis(
                std::make_shared<_IRBasis<sparseir::Fermionic>>(impl));
        } else {
            using FiniteTempBasisType = sparseir::FiniteTempBasis<sparseir::Bosonic>;
            auto impl = std::make_shared<FiniteTempBasisType>(
                beta, omega_max, epsilon, lambda, ypower, conv_radius,
                *sve_impl, inv_weight_func, max_size);
            result = create_basis(
                std::make_shared<_IRBasis<sparseir::Bosonic>>(impl));
        }

        if (result) {
            *status = SPIR_COMPUTATION_SUCCESS;
            return result;
        } else {
            DEBUG_LOG("Failed to create basis: create_basis returned nullptr");
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_new_from_sve_and_inv_weight: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_new_from_sve_and_inv_weight");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_basis* spir_basis_new(
    int statistics, double beta, double omega_max, double epsilon,
    const spir_kernel *k, const spir_sve_result *sve, int max_size, int* status)
{
    try {
        // Get the kernel implementation
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get a basis implementation");
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // switch on kernel type
        spir_basis* result = nullptr;
        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            result = _spir_basis_new(statistics, beta, omega_max, epsilon, *logistic, sve, max_size);
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            result = _spir_basis_new(statistics, beta, omega_max, epsilon, *bose, sve, max_size);
        } else {
            DEBUG_LOG("Unknown kernel type");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        if (!result) {
            DEBUG_LOG("Failed to create finite temperature basis");
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        *status = SPIR_COMPUTATION_SUCCESS;
        return result;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_basis_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}


spir_sampling* spir_tau_sampling_new(const spir_basis *b, int num_points, const double *points, int* status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_tau_sampling_new: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!points) {
        DEBUG_LOG("Error in spir_tau_sampling_new: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_tau_sampling_new: num_points is less than or equal to 0");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        DEBUG_LOG("Failed to get a tau sampling implementation");
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            *status = SPIR_COMPUTATION_SUCCESS;
            return _spir_tau_sampling_new_with_points<sparseir::TauSampling<sparseir::Fermionic>>(
                b, num_points, points);
        } else {
            *status = SPIR_COMPUTATION_SUCCESS;
            return _spir_tau_sampling_new_with_points<sparseir::TauSampling<sparseir::Bosonic>>(
                b, num_points, points);
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_tau_sampling_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_sampling* spir_tau_sampling_new_with_matrix(int order, int statistics, int basis_size, int num_points, const double *points, const double *matrix, int* status)
{
    if (!points) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!matrix) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: invalid pointer matrix");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: num_points is less than or equal to 0");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

        // check statistics
    if (statistics != SPIR_STATISTICS_FERMIONIC && statistics != SPIR_STATISTICS_BOSONIC) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid statistics");
        return nullptr;
    }

    // check order
    if (order != SPIR_ORDER_ROW_MAJOR && order != SPIR_ORDER_COLUMN_MAJOR) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid order");
        return nullptr;
    }

    try {
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            return _spir_tau_sampling_new_with_matrix<sparseir::TauSampling<sparseir::Fermionic>>(
                order, basis_size, num_points, points, matrix, status);
        } else {
            return _spir_tau_sampling_new_with_matrix<sparseir::TauSampling<sparseir::Bosonic>>(
                order, basis_size, num_points, points, matrix, status);
        }
    } catch (const std::exception& e) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_sampling* spir_matsu_sampling_new(const spir_basis *b, bool positive_only, int num_points, const int64_t *points, int* status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_matsu_sampling_new: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!points) {
        DEBUG_LOG("Error in spir_matsu_sampling_new: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_matsu_sampling_new: num_points is less than or equal to 0");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    if (positive_only) {
        for (int i = 0; i < num_points; i++) {
            if (points[i] < 0) {
                DEBUG_LOG("Error: Frequency must not be negative if positive_only is true");
                *status = SPIR_INVALID_ARGUMENT;
                return nullptr;
            }
        }
    }

    // Get basis functions
    spir_funcs* uhat = spir_basis_get_uhat(b, status);
    if (!uhat) {
        DEBUG_LOG("Error: Failed to get basis functions");
        return nullptr;
    }

    // Get basis size
    int basis_size;
    int size_status = spir_basis_get_size(b, &basis_size);
    if (size_status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to get basis size");
        spir_funcs_release(uhat);
        *status = size_status;
        return nullptr;
    }

    int statistics;
    int stats_status = spir_basis_get_stats(b, &statistics);
    if (stats_status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to get basis statistics");
        spir_funcs_release(uhat);
        *status = stats_status;
        return nullptr;
    }

    // Allocate memory for matrix
    std::vector<std::complex<double>> matrix(num_points * basis_size);
    int eval_status = spir_funcs_batch_eval_matsu(uhat, SPIR_ORDER_COLUMN_MAJOR, num_points, points, (c_complex*)matrix.data());
    if (eval_status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to evaluate basis functions");
        spir_funcs_release(uhat);
        *status = eval_status;
        return nullptr;
    }

    // Create sampling object using the matrix version
    spir_sampling* smpl = spir_matsu_sampling_new_with_matrix(
        SPIR_ORDER_COLUMN_MAJOR, statistics, basis_size, positive_only, num_points, points, (c_complex*)matrix.data(), status);

    spir_funcs_release(uhat);

    return smpl;
}

spir_sampling* spir_matsu_sampling_new_with_matrix(
                                                  int order,
                                                  int statistics,
                                                  int basis_size,
                                                  bool positive_only,
                                                  int num_points,
                                                  const int64_t *points,
                                                  const c_complex *matrix,
                                                  int *status)
{
    if (!points) {
        DEBUG_LOG("Error in spir_matsu_sampling_new_with_matrix: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!matrix) {
        DEBUG_LOG("Error in spir_matsu_sampling_new_with_matrix: invalid pointer matrix");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_matsu_sampling_new_with_matrix: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    // check statistics
    if (statistics != SPIR_STATISTICS_FERMIONIC && statistics != SPIR_STATISTICS_BOSONIC) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid statistics");
        return nullptr;
    }

    // check order
    if (order != SPIR_ORDER_ROW_MAJOR && order != SPIR_ORDER_COLUMN_MAJOR) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid order");
        return nullptr;
    }

    if (statistics == SPIR_STATISTICS_FERMIONIC) {
        return _spir_matsu_sampling_new_with_matrix<sparseir::Fermionic, sparseir::MatsubaraSampling<sparseir::Fermionic>>(
            order, basis_size, positive_only, num_points, points, matrix, status);
    } else {
        return _spir_matsu_sampling_new_with_matrix<sparseir::Bosonic, sparseir::MatsubaraSampling<sparseir::Bosonic>>(
            order, basis_size, positive_only, num_points, points, matrix, status);
    }
}

spir_basis* spir_dlr_new(const spir_basis *b, int* status)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    int stat;
    int status_basis = spir_basis_get_stats(b, &stat);
    if (status_basis != SPIR_COMPUTATION_SUCCESS) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new<sparseir::Fermionic>(b);
    } else {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new<sparseir::Bosonic>(b);
    }
}

spir_basis* spir_dlr_new_with_poles(const spir_basis *b,
                                  const int npoles, const double *poles, int* status)
{
    auto impl = get_impl_basis(b);
    if (!impl)
        return nullptr;

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    int stat;
    int status_basis = spir_basis_get_stats(b, &stat);
    if (status_basis != SPIR_COMPUTATION_SUCCESS) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new_with_poles<sparseir::Fermionic>(b, npoles, poles);
    } else {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new_with_poles<sparseir::Bosonic>(b, npoles, poles);
    }
}

int spir_sampling_eval_dd(const spir_sampling *s, int order,
                                  int ndim, const int *input_dims,
                                  int target_dim, const double *input,
                                  double *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                         &sparseir::AbstractSampling::evaluate_inplace_dd);
}

int spir_sampling_eval_dz(const spir_sampling *s, int order,
                                  int ndim, const int *input_dims,
                                  int target_dim, const double *input,
                                  c_complex *out)
{
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, cpp_out,
                         &sparseir::AbstractSampling::evaluate_inplace_dz);
}

int spir_sampling_eval_zz(const spir_sampling *s, int order,
                                  int ndim, const int *input_dims,
                                  int target_dim, const c_complex *input,
                                  c_complex *out)
{
    // DANGER: MEMORY LAYOUT MAY NOT BE CONSISTENT BETWEEN C99 AND C++
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return evaluate_impl(s, order, ndim, input_dims, target_dim, cpp_input,
                         cpp_out,
                         &sparseir::AbstractSampling::evaluate_inplace_zz);
}

int spir_sampling_fit_dd(const spir_sampling *s, int order,
                             int ndim, const int *input_dims,
                             int target_dim, const double *input,
                             double *out)
{
    return fit_impl(s, order, ndim, input_dims, target_dim, input, out,
                    &sparseir::AbstractSampling::fit_inplace_dd);
}

int spir_sampling_fit_zz(const spir_sampling *s, int order,
                             int ndim, const int *input_dims,
                             int target_dim, const c_complex *input,
                             c_complex *out)
{
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return fit_impl(s, order, ndim, input_dims, target_dim, cpp_input, cpp_out,
                    &sparseir::AbstractSampling::fit_inplace_zz);
}

int spir_sampling_fit_zd(const spir_sampling *s, int order,
                             int ndim, const int *input_dims,
                             int target_dim, const c_complex *input,
                             double *out)
{
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    return fit_impl(s, order, ndim, input_dims, target_dim, cpp_input, out,
                    &sparseir::AbstractSampling::fit_inplace_zd);
}

int spir_dlr2ir_dd(const spir_basis *dlr, int order, int ndim,
                       const int *input_dims, int target_dim,
                       const double *input, double *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl) {
        DEBUG_LOG("Error in spir_dlr2ir_dd: failed to get basis implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr2ir<sparseir::Fermionic, double>(dlr, order, ndim, input_dims, target_dim,
                                                   input, out);
    } else {
        return spir_dlr2ir<sparseir::Bosonic, double>(dlr, order, ndim, input_dims, target_dim,
                                                 input, out);
    }
}

int spir_dlr2ir_zz(const spir_basis *dlr, int order, int ndim,
                       const int *input_dims, int target_dim,
                       const c_complex *input, c_complex *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr2ir<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                   cpp_input, cpp_out);
    } else {
        return spir_dlr2ir<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                 cpp_input, cpp_out);
    }
}

int spir_ir2dlr_dd(const spir_basis *dlr, int order,
                         int ndim, const int *input_dims, int target_dim,
                         const double *input, double *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_ir2dlr<sparseir::Fermionic, double>(dlr, order, ndim,
                                                     input_dims, target_dim, input, out);
    } else {
        return spir_ir2dlr<sparseir::Bosonic, double>(dlr, order, ndim, input_dims,
                                                   target_dim, input, out);
    }
}

int spir_ir2dlr_zz(const spir_basis *dlr, int order,
                         int ndim, const int *input_dims, int target_dim,
                         const c_complex *input, c_complex *out)
{
    auto impl = get_impl_basis(dlr);
    if  (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_ir2dlr<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim,
                                                     input_dims, target_dim, cpp_input, cpp_out);
    } else {
        return spir_ir2dlr<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims,
                                                   target_dim, cpp_input, cpp_out);
    }
}

int spir_dlr_get_npoles(const spir_basis *dlr, int *num_poles)
{
    if (!dlr || !num_poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        *num_poles = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_dlr_get_poles(const spir_basis *dlr, double *poles)
{
    if (!dlr || !poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        std::vector<double> poles_vec = std::dynamic_pointer_cast<AbstractDLR>(impl)->get_poles();
        std::memcpy(poles, poles_vec.data(), poles_vec.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

spir_funcs* spir_basis_get_u(const spir_basis *b, int *status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_u: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_basis_get_u: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            spir_funcs *u = nullptr;
            int ret = _spir_basis_get_u<sparseir::Fermionic>(b, &u);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return u;
        } else {
            spir_funcs *u = nullptr;
            int ret = _spir_basis_get_u<sparseir::Bosonic>(b, &u);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return u;
        }
    } catch (const std::exception &e) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
}

spir_funcs* spir_basis_get_v(const spir_basis *b, int *status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_v: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_basis_get_v: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            spir_funcs *v = nullptr;
            int ret = _spir_get_v<sparseir::Fermionic>(b, &v);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return v;
        } else {
            spir_funcs *v = nullptr;
            int ret = _spir_get_v<sparseir::Bosonic>(b, &v);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return v;
        }
    } catch (const std::exception &e) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
}

spir_funcs* spir_basis_get_uhat(const spir_basis *b, int *status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_uhat: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_basis_get_uhat: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            spir_funcs *uhat = nullptr;
            int ret = _spir_basis_get_uhat<sparseir::Fermionic>(b, &uhat);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return uhat;
        } else {
            spir_funcs *uhat = nullptr;
            int ret = _spir_basis_get_uhat<sparseir::Bosonic>(b, &uhat);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return uhat;
        }
    } catch (const std::exception &e) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
}

spir_funcs* spir_basis_get_uhat_full(const spir_basis *b, int *status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_uhat_full: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_basis_get_uhat_full: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            spir_funcs *uhat_full = nullptr;
            int ret = _spir_basis_get_uhat_full<sparseir::Fermionic>(b, &uhat_full);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return uhat_full;
        } else {
            spir_funcs *uhat_full = nullptr;
            int ret = _spir_basis_get_uhat_full<sparseir::Bosonic>(b, &uhat_full);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return uhat_full;
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_uhat_full: " + std::string(e.what()));
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
}

// TODO: USE THIS
int spir_sampling_get_npoints(const spir_sampling *s,
                                     int *num_points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        DEBUG_LOG("Error in spir_sampling_get_npoints: failed to get sampling implementation");
        return SPIR_GET_IMPL_FAILED;
    }
    if (!num_points) {
        DEBUG_LOG("Error in spir_sampling_get_npoints: invalid pointer num_points");
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *num_points = impl->n_sampling_points();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        DEBUG_LOG("Error in spir_sampling_get_npoints: unknown exception");
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_sampling_get_taus(const spir_sampling *s, double *points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        DEBUG_LOG("Error in spir_sampling_get_taus: failed to get sampling implementation");
        return SPIR_GET_IMPL_FAILED;
    }
    try {
        // Try Fermionic case first
        auto fermionic_sampling = std::dynamic_pointer_cast<
            sparseir::TauSampling<sparseir::Fermionic>>(impl);
        if (fermionic_sampling) {
            auto tau_points = fermionic_sampling->sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try Bosonic case
        auto bosonic_sampling =
            std::dynamic_pointer_cast<sparseir::TauSampling<sparseir::Bosonic>>(
                impl);
        if (bosonic_sampling) {
            auto tau_points = bosonic_sampling->sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }

        return SPIR_NOT_SUPPORTED;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

// TODO: USE THIS
int spir_sampling_get_matsus(const spir_sampling *s,
                                           int64_t *points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    try {
        // Try Fermionic case first
        auto fermionic_sampling = std::dynamic_pointer_cast<
            sparseir::MatsubaraSampling<sparseir::Fermionic>>(impl);
        if (fermionic_sampling) {
            auto matsubara_points = fermionic_sampling->sampling_points();
            std::transform(
                matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Fermionic> &freq) {
                    return static_cast<int64_t>(freq.get_n());
                });
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try Bosonic case
        auto bosonic_sampling = std::dynamic_pointer_cast<
            sparseir::MatsubaraSampling<sparseir::Bosonic>>(impl);
        if (bosonic_sampling) {
            auto matsubara_points = bosonic_sampling->sampling_points();
            std::transform(
                matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Bosonic> &freq) {
                    return static_cast<int64_t>(freq.get_n());
                });
            return SPIR_COMPUTATION_SUCCESS;
        }

        return SPIR_NOT_SUPPORTED;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_size(const spir_basis *b,
                                        int *size)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        DEBUG_LOG("Error in spir_basis_get_size: failed to get basis implementation");
        return SPIR_GET_IMPL_FAILED;
    }
    if (!size) {
        DEBUG_LOG("Error in spir_basis_get_size: invalid pointer size");
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *size = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        DEBUG_LOG("Error in spir_basis_get_size: unknown exception");
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_n_default_taus(const spir_basis *b, int *num_points)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_n_default_taus: invalid pointer b");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!num_points) {
        DEBUG_LOG("Error in spir_basis_get_n_default_taus: invalid pointer num_points");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto points = ir_basis->default_tau_sampling_points();
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto points = ir_basis->default_tau_sampling_points();
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_taus(const spir_basis *b, double *points)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_default_taus: invalid pointer b");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!points) {
        DEBUG_LOG("Error in spir_basis_get_default_taus: invalid pointer points");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto tau_points = ir_basis->default_tau_sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto tau_points = ir_basis->default_tau_sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_taus_ext(
    const spir_basis *b, int n_points, double *points, int *n_points_returned)
{
    if (!b || !points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    double beta = impl->get_beta();

    try {
        Eigen::VectorXd tau_points;
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            // Use default_tau_sampling_points() method which handles augmentation correctly
            // But we need to request n_points, not size()
            int sz = ir_basis->get_impl()->size();
            Eigen::VectorXd x = sparseir::default_sampling_points(
                *(ir_basis->get_impl()->sve_result->u), n_points
            );

            // Process like default_tau_sampling_points() but with n_points
            std::vector<double> unique_x;
            if (x.size() % 2 == 0) {
                for (auto i = 0; i < x.size() / 2; ++i) {
                    unique_x.push_back(x(i));
                }
            } else {
                for (auto i = 0; i < x.size() / 2; ++i) {
                    unique_x.push_back(x(i));
                }
                auto x_new = 0.5 * (unique_x.back() + 0.5);
                unique_x.push_back(x_new);
            }

            tau_points = Eigen::VectorXd(2 * unique_x.size());
            double beta = impl->get_beta();
            for (auto i = 0; i < unique_x.size(); ++i) {
                tau_points(i) = (beta / 2.0) * (unique_x[i] + 1.0);
                tau_points(unique_x.size() + i) = -tau_points(i);
            }
            std::sort(tau_points.data(), tau_points.data() + tau_points.size());
            *n_points_returned = std::min(n_points, static_cast<int>(tau_points.size()));
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            // Use default_tau_sampling_points() method which handles augmentation correctly
            // But we need to request n_points, not size()
            int sz = ir_basis->get_impl()->size();
            Eigen::VectorXd x = sparseir::default_sampling_points(
                *(ir_basis->get_impl()->sve_result->u), n_points
            );

            // Process like default_tau_sampling_points() but with n_points
            std::vector<double> unique_x;
            if (x.size() % 2 == 0) {
                for (auto i = 0; i < x.size() / 2; ++i) {
                    unique_x.push_back(x(i));
                }
            } else {
                for (auto i = 0; i < x.size() / 2; ++i) {
                    unique_x.push_back(x(i));
                }
                auto x_new = 0.5 * (unique_x.back() + 0.5);
                unique_x.push_back(x_new);
            }

            tau_points = Eigen::VectorXd(2 * unique_x.size());
            double beta = impl->get_beta();
            for (auto i = 0; i < unique_x.size(); ++i) {
                tau_points(i) = (beta / 2.0) * (unique_x[i] + 1.0);
                tau_points(unique_x.size() + i) = -tau_points(i);
            }
            std::sort(tau_points.data(), tau_points.data() + tau_points.size());
            *n_points_returned = std::min(n_points, static_cast<int>(tau_points.size()));
        }

        // Copy the requested number of points (or available points if less)
        std::copy(tau_points.data(), tau_points.data() + *n_points_returned, points);
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_n_default_matsus(const spir_basis *b, bool positive_only, int *num_points)
{
    if (!b || !num_points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        DEBUG_LOG("Error: Failed to get basis implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points(positive_only);
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points(positive_only);
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Error in spir_basis_get_n_default_matsus: " + std::string(e.what()));
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_matsus(const spir_basis *b, bool positive_only, int64_t *points)
{
    if (!b || !points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto matsubara_points = ir_basis->default_matsubara_sampling_points(positive_only);
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto matsubara_points = ir_basis->default_matsubara_sampling_points(positive_only);
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_n_default_matsus_ext(const spir_basis *b, bool positive_only, int L, int *num_points_returned)
{
    if (!b || !num_points_returned) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points_ext(L, positive_only);
            *num_points_returned = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points_ext(L, positive_only);
            *num_points_returned = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_matsus_ext(const spir_basis *b, bool positive_only, bool mitigate, int n_points, int64_t *points, int *n_points_returned)
{
    if (!b || !points || !n_points_returned) {
        DEBUG_LOG("Error: Invalid arguments in spir_basis_get_default_matsus_ext");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        DEBUG_LOG("Error: Failed to get basis implementation in spir_basis_get_default_matsus_ext");
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis in spir_basis_get_default_matsus_ext");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        DEBUG_LOG("spir_basis_get_default_matsus_ext: statistics=" + std::to_string(impl->get_statistics()) + ", n_points=" + std::to_string(n_points) + ", positive_only=" + std::to_string(positive_only) + ", mitigate=" + std::to_string(mitigate));
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            DEBUG_LOG("spir_basis_get_default_matsus_ext: casting to Fermionic");
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            DEBUG_LOG("spir_basis_get_default_matsus_ext: calling default_matsubara_sampling_points_ext");
            auto matsubara_points = ir_basis->default_matsubara_sampling_points_ext(n_points, positive_only, mitigate);
            DEBUG_LOG("spir_basis_get_default_matsus_ext: got " + std::to_string(matsubara_points.size()) + " points");
            *n_points_returned = matsubara_points.size();
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            DEBUG_LOG("spir_basis_get_default_matsus_ext: casting to Bosonic");
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            DEBUG_LOG("spir_basis_get_default_matsus_ext: calling default_matsubara_sampling_points_ext");
            auto matsubara_points = ir_basis->default_matsubara_sampling_points_ext(n_points, positive_only, mitigate);
            DEBUG_LOG("spir_basis_get_default_matsus_ext: got " + std::to_string(matsubara_points.size()) + " points");
            *n_points_returned = matsubara_points.size();
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_default_matsus_ext: " + std::string(e.what()));
        return SPIR_GET_IMPL_FAILED;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_get_default_matsus_ext");
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_stats(const spir_basis *b,
                                  int *statistics)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (!statistics) {
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *statistics = impl->get_statistics();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_uhat_get_default_matsus(const spir_funcs *uhat, int L, bool positive_only, bool mitigate, int64_t *points, int *n_points_returned)
{
    if (!uhat || !points || !n_points_returned) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(uhat);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    // Check if this is a MatsubaraBasisFunctions (not continuous functions)
    if (impl->is_continuous_funcs()) {
        DEBUG_LOG("Error: uhat must be a MatsubaraBasisFunctions (PiecewiseLegendreFTVector)");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        // Cast to AbstractMatsubaraFunctions
        auto matsubara_impl = std::dynamic_pointer_cast<AbstractMatsubaraFunctions>(impl);
        if (!matsubara_impl) {
            DEBUG_LOG("Error: uhat is not a MatsubaraBasisFunctions");
            return SPIR_INVALID_ARGUMENT;
        }

        // Try to cast to MatsubaraBasisFunctions<PiecewiseLegendreFTVector<Fermionic>>
        auto fermionic_uhat = std::dynamic_pointer_cast<MatsubaraBasisFunctions<sparseir::PiecewiseLegendreFTVector<sparseir::Fermionic>>>(matsubara_impl);
        if (fermionic_uhat) {
            auto uhat_impl = fermionic_uhat->get_impl();
            bool fence = mitigate;
            std::vector<sparseir::MatsubaraFreq<sparseir::Fermionic>> matsubara_points =
                sparseir::default_matsubara_sampling_points_impl(*uhat_impl, L, fence, positive_only);
            *n_points_returned = matsubara_points.size();
            std::transform(
                matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Fermionic> &freq) {
                    return static_cast<int64_t>(freq.get_n());
                });
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try to cast to MatsubaraBasisFunctions<PiecewiseLegendreFTVector<Bosonic>>
        auto bosonic_uhat = std::dynamic_pointer_cast<MatsubaraBasisFunctions<sparseir::PiecewiseLegendreFTVector<sparseir::Bosonic>>>(matsubara_impl);
        if (bosonic_uhat) {
            auto uhat_impl = bosonic_uhat->get_impl();
            bool fence = mitigate;
            std::vector<sparseir::MatsubaraFreq<sparseir::Bosonic>> matsubara_points =
                sparseir::default_matsubara_sampling_points_impl(*uhat_impl, L, fence, positive_only);
            *n_points_returned = matsubara_points.size();
            std::transform(
                matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Bosonic> &freq) {
                    return static_cast<int64_t>(freq.get_n());
                });
            return SPIR_COMPUTATION_SUCCESS;
        }

        DEBUG_LOG("Error: uhat is not a PiecewiseLegendreFTVector (Fermionic or Bosonic)");
        return SPIR_INVALID_ARGUMENT;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_uhat_get_default_matsus: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_uhat_get_default_matsus");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_eval(const spir_funcs *funcs, double x, double *out)
{
    if (!out) {
        DEBUG_LOG("Error in spir_funcs_eval: out is null");
        return SPIR_INVALID_ARGUMENT;
    }
    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (!impl->is_continuous_funcs()) {
        DEBUG_LOG("Error: the function is not defined for continuous variables");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        Eigen::VectorXd result = std::dynamic_pointer_cast<AbstractContinuousFunctions>(funcs->ptr)->operator()(x);
        std::memcpy(out, result.data(), result.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_eval: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_eval_matsu(const spir_funcs *funcs, int64_t x, c_complex *out)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_eval_matsu: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!out) {
        DEBUG_LOG("Error in spir_funcs_eval_matsu: invalid pointer out");
        return SPIR_INVALID_ARGUMENT;
    }

    // Use batch_evaluate_matsubara with num_freqs = 1
    return spir_funcs_batch_eval_matsu(funcs, SPIR_ORDER_COLUMN_MAJOR, 1, &x, out);
}

int spir_funcs_batch_eval(const spir_funcs *funcs,
                                 int order, int num_points,
                                 const double *xs, double *out)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!xs) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: invalid pointer xs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!out) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: invalid pointer out");
        return SPIR_INVALID_ARGUMENT;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: num_points is less than or equal to 0");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        // Get the size of the functions object
        int size;
        int status = spir_funcs_get_size(funcs, &size);
        if (status != SPIR_COMPUTATION_SUCCESS) {
            return status;
        }

        // result is a matrix of size n_funcs x num_points in column-major order
        Eigen::MatrixXd result = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl)->operator()(Eigen::Map<const Eigen::VectorXd>(xs, num_points));

        // out is a matrix of size num_points x n_funcs
        if (order == SPIR_ORDER_ROW_MAJOR) {
            // Copy the results to the output array
            for (int i = 0; i < num_points; ++i) {
                for (int j = 0; j < size; ++j) {
                    out[i * size + j] = result(j, i);
                }
            }
        } else {
            // Copy the results to the output array
            for (int i = 0; i < num_points; ++i) {
                for (int j = 0; j < size; ++j) {
                    out[j * num_points + i] = result(j, i);
                }
            }
        }

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_batch_eval: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_batch_eval");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_batch_eval_matsu(const spir_funcs *uiw,
                                          int order,
                                          int num_freqs,
                                          const int64_t *matsubara_freq_indices,
                                          c_complex *out)
{
    auto impl = get_impl_funcs(uiw);
    if (!impl) {
        DEBUG_LOG("Matsubara basis functions object is null or not assigned");
        return SPIR_GET_IMPL_FAILED;
    }
    if (impl->is_continuous_funcs()) {
        DEBUG_LOG("Error: the function is not defined for Matsubara frequencies");
        return SPIR_INVALID_ARGUMENT;
    }

    if (!matsubara_freq_indices || !out) {
        DEBUG_LOG("Input or output array is null");
        return SPIR_INVALID_ARGUMENT;
    }

    if (num_freqs <= 0) {
        DEBUG_LOG("Number of frequencies must be positive");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        // Convert C array to Eigen vector
        Eigen::Vector<int64_t, Eigen::Dynamic> freq_indices =
            Eigen::Map<const Eigen::Vector<int64_t, Eigen::Dynamic>>(matsubara_freq_indices, num_freqs);

        // Get the func size
        int func_size = uiw->ptr->size();

        // Evaluate functions at all frequencies
        // The operator() returns a matrix of shape (nfuncs, nfreqs)
        Eigen::MatrixXcd out_matrix = std::dynamic_pointer_cast<AbstractMatsubaraFunctions>(uiw->ptr)->operator()(freq_indices);

        // Copy the results to the output array
        for (int ifreq = 0; ifreq < num_freqs; ++ifreq) {
            for (int ifunc = 0; ifunc < func_size; ++ifunc) {
                const auto &val = out_matrix(ifunc, ifreq);
                if (order == SPIR_ORDER_ROW_MAJOR) {
                    out[ifreq * func_size + ifunc] = {val.real(), val.imag()};
                } else {
                    out[ifunc * num_freqs + ifreq] = {val.real(), val.imag()};
                }
            }
        }

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_batch_eval_matsu: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_size(const spir_funcs *funcs, int *size)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_size: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!size) {
        DEBUG_LOG("Error in spir_funcs_get_size: invalid pointer size");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        auto impl = get_impl_funcs(funcs);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        *size = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

spir_funcs *spir_funcs_get_slice(const spir_funcs *funcs, int nslice, int *indices, int *status)
{
    try {
        // Get the implementation
        auto impl = get_impl_funcs(funcs);
        if (!impl) {
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // Convert indices to vector<size_t>
        std::vector<size_t> indices_vec(indices, indices + nslice);

        // Check for duplicates and out of range
        try {
            check_indices(indices_vec, impl->size());
        } catch (const std::runtime_error& e) {
            DEBUG_LOG("Error: " + std::string(e.what()));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Get the slice
        auto sliced_impl = impl->slice(indices_vec);
        if (!sliced_impl) {
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        // Create new funcs object
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_funcs(sliced_impl);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_slice: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int spir_basis_get_n_default_ws(const spir_basis *b,
                                                       int *num_points)
{
    if (!b || !num_points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto points = basis->default_omega_sampling_points();
            *num_points = points.size();
        } else {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto points = basis->default_omega_sampling_points();
            *num_points = points.size();
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_n_default_ws: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_get_n_default_ws");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_basis_get_default_ws(const spir_basis *b,
                                                   double *points)
{
    if (!b || !points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto sampling_points = basis->default_omega_sampling_points();
            std::copy(sampling_points.begin(), sampling_points.end(), points);
        } else {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto sampling_points = basis->default_omega_sampling_points();
            std::copy(sampling_points.begin(), sampling_points.end(), points);
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_default_ws: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_get_default_ws");
        return SPIR_INTERNAL_ERROR;
    }
}

void *_spir_basis_get_raw_ptr(const spir_basis *obj)
{
    if (!obj) {
        return nullptr;
    }
    auto impl = get_impl_basis(obj);
    if (!impl) {
        return nullptr;
    }
    return impl.get();
}

int spir_sve_result_get_size(const spir_sve_result *sve, int *size)
{
    if (!sve || !size) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_sve_result(sve);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        *size = impl->s.size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_get_size: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_get_size");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_sve_result_get_svals(const spir_sve_result *sve, double *svals)
{
    if (!sve || !svals) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_sve_result(sve);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        std::memcpy(svals, impl->s.data(), impl->s.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_get_svals: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_get_svals");
        return SPIR_INTERNAL_ERROR;
    }
}

spir_sve_result* spir_sve_result_truncate(const spir_sve_result *sve, double epsilon, int max_size, int *status)
{
    if (!sve || !status) {
        if (status) *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_sve_result(sve);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        // Use the part method to truncate the SVE result
        auto part_result = impl->part(epsilon, max_size);

        // Extract the truncated components
        sparseir::PiecewiseLegendrePolyVector u_truncated = std::get<0>(part_result);
        Eigen::VectorXd s_truncated = std::get<1>(part_result);
        sparseir::PiecewiseLegendrePolyVector v_truncated = std::get<2>(part_result);

        // Create a new SVEResult with the truncated components
        // Use the original epsilon or the provided one if it's not NaN
        double result_epsilon = std::isnan(epsilon) ? impl->epsilon : epsilon;
        auto truncated_sve_result = std::make_shared<sparseir::SVEResult>(
            u_truncated, s_truncated, v_truncated, result_epsilon);

        *status = SPIR_COMPUTATION_SUCCESS;
        return create_sve_result(truncated_sve_result);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_truncate: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_truncate");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int spir_basis_get_svals(const spir_basis *b, double *svals)
{
    if (!b || !svals) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        std::cerr << "Error: The basis is not an IR basis" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto basis = _safe_dynamic_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto s = basis->s();
            std::memcpy(svals, s.data(), s.size() * sizeof(double));
        } else {
            auto basis = _safe_dynamic_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto s = basis->s();
            std::memcpy(svals, s.data(), s.size() * sizeof(double));
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_svals: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_get_svals");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_n_roots(const spir_funcs *funcs, int *n_roots)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_n_roots: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!n_roots) {
        DEBUG_LOG("Error in spir_funcs_get_n_roots: invalid pointer n_roots");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        *n_roots = continuous_impl->nroots();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_n_roots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_n_roots");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_roots(const spir_funcs *funcs, double *roots)
{
    if (!funcs || !roots) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        // Get the roots from the implementation
        Eigen::VectorXd roots_vec = continuous_impl->roots();

        // Copy the roots to the output array
        std::memcpy(roots, roots_vec.data(), roots_vec.size() * sizeof(double));

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_roots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_roots");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_sampling_get_cond_num(const spir_sampling *s, double *cond_num)
{
    DEBUG_LOG("spir_sampling_get_cond_num called with sampling=" + std::to_string(reinterpret_cast<uintptr_t>(s)));
    auto impl = get_impl_sampling(s);
    if (!impl) {
        DEBUG_LOG("Failed to get sampling implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!cond_num) {
            DEBUG_LOG("cond_num is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }

        *cond_num = impl->get_cond_num();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sampling_get_cond_num: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sampling_get_cond_num");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_n_knots(const spir_funcs *funcs, int *n_knots)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_n_knots: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!n_knots) {
        DEBUG_LOG("Error in spir_funcs_get_n_knots: invalid pointer n_knots");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            DEBUG_LOG("Error: the function is not defined for continuous variables");
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        // Check if this is a PiecewiseLegendrePolyFunctions or TauFunctionsAdaptor
        auto piecewise_impl = std::dynamic_pointer_cast<PiecewiseLegendrePolyFunctions>(continuous_impl);
        if (piecewise_impl) {
            // Get knots from the underlying PiecewiseLegendrePolyVector
            auto knots = piecewise_impl->get_impl()->get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try TauFunctionsAdaptor
        auto tau_adaptor = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Fermionic>>>(continuous_impl);
        if (tau_adaptor) {
            auto knots = tau_adaptor->get_impl()->get_obj().get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        }

        auto tau_adaptor_boson = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Bosonic>>>(continuous_impl);
        if (tau_adaptor_boson) {
            auto knots = tau_adaptor_boson->get_impl()->get_obj().get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try OmegaFunctionsAdaptor - use the actual type from the implementation
        // We need to check what type the OmegaFunctionsAdaptor is wrapping
        // For now, just call get_knots() directly on the continuous_impl
        try {
            auto knots = continuous_impl->get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        } catch (...) {
            // If get_knots() is not implemented, return empty
            *n_knots = 0;
            return SPIR_COMPUTATION_SUCCESS;
        }

        DEBUG_LOG("Error: knots are only available for piecewise Legendre polynomial functions");
        return SPIR_NOT_SUPPORTED;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_n_knots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_n_knots");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_knots(const spir_funcs *funcs, double *knots)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_knots: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!knots) {
        DEBUG_LOG("Error in spir_funcs_get_knots: invalid pointer knots");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            DEBUG_LOG("Error: the function is not defined for continuous variables");
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        // Check if this is a PiecewiseLegendrePolyFunctions or TauFunctionsAdaptor
        auto piecewise_impl = std::dynamic_pointer_cast<PiecewiseLegendrePolyFunctions>(continuous_impl);
        if (piecewise_impl) {
            // Get knots from the underlying PiecewiseLegendrePolyVector
            auto knots_vec = piecewise_impl->get_impl()->get_knots();

            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try TauFunctionsAdaptor
        auto tau_adaptor = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Fermionic>>>(continuous_impl);
        if (tau_adaptor) {
            auto knots_vec = tau_adaptor->get_impl()->get_obj().get_knots();

            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        }

        auto tau_adaptor_boson = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Bosonic>>>(continuous_impl);
        if (tau_adaptor_boson) {
            auto knots_vec = tau_adaptor_boson->get_impl()->get_obj().get_knots();

            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try OmegaFunctionsAdaptor - use the actual type from the implementation
        // We need to check what type the OmegaFunctionsAdaptor is wrapping
        // For now, just call get_knots() directly on the continuous_impl
        try {
            auto knots_vec = continuous_impl->get_knots();

            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        } catch (...) {
            // If get_knots() is not implemented, return success with empty array
            return SPIR_COMPUTATION_SUCCESS;
        }

        DEBUG_LOG("Error: knots are only available for piecewise Legendre polynomial functions");
        return SPIR_NOT_SUPPORTED;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_knots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_knots");
        return SPIR_INTERNAL_ERROR;
    }
}

} // extern "C"
