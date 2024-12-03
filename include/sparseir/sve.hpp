// Include necessary headers
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <limits>
#include <cmath>
#include <iostream>
#include <numeric>

// Namespace
namespace sparseir {

// Function templates
template<typename T>
inline T sqrt_eps() {
    return std::sqrt(std::numeric_limits<T>::epsilon());
}

// Helper function for sign
inline double sign(double x) {
    return (x > 0) - (x < 0);
}

// SVEResult class
template <typename T>
class SVEResult {
public:
    // Assume PiecewiseLegendrePolyVector is a user-defined type representing the polynomials
    // For this example, we'll use vectors of Eigen matrices as placeholders
    std::vector<Eigen::MatrixX<T>> u; // Left singular functions
    Eigen::VectorX<T> s; // Singular values
    std::vector<Eigen::MatrixX<T>> v; // Right singular functions

    AbstractKernel* kernel;
    double epsilon;

    SVEResult(const std::vector<Eigen::MatrixX<T>>& u_,
              const Eigen::VectorX<T>& s_,
              const std::vector<Eigen::MatrixX<T>>& v_,
              AbstractKernel* K,
              double eps)
        : u(u_), s(s_), v(v_), kernel(K), epsilon(eps) {}

    std::tuple<std::vector<Eigen::MatrixX<T>>, Eigen::VectorX<T>, std::vector<Eigen::MatrixX<T>>>
    part(double eps_input = -1, int max_size = -1) const {
        double eps_threshold = (eps_input == -1) ? epsilon : eps_input;
        int cut = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (s(i) >= eps_threshold * s(0)) {
                ++cut;
            }
        }
        if (max_size != -1 && max_size < cut) {
            cut = max_size;
        }
        if (cut == s.size()) {
            return std::make_tuple(u, s, v);
        } else {
            std::vector<Eigen::MatrixX<T>> u_cut(u.begin(), u.begin() + cut);
            Eigen::VectorX<T> s_cut = s.head(cut);
            std::vector<Eigen::MatrixX<T>> v_cut(v.begin(), v.begin() + cut);
            return std::make_tuple(u_cut, s_cut, v_cut);
        }
    }
};

// AbstractSVE class
template <typename T>
class AbstractSVE
{
public:
    virtual ~AbstractSVE() {}
    virtual std::vector<Eigen::MatrixX<T>> matrices() const = 0;
    virtual SVEResult<T> postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                                  const std::vector<Eigen::VectorX<T>> &s_list,
                                  const std::vector<Eigen::MatrixX<T>> &v_list) = 0;
    int nsvals_hint;
};

// SamplingSVE class
template <typename T>
class SamplingSVE : AbstractSVE<T>{
public:
    AbstractKernel* kernel;
    double epsilon;
    int n_gauss;
    int nsvals_hint;

    // Additional members to store quadrature rules, segments, etc.
    // For simplicity, we'll use Eigen types to represent them
    Eigen::VectorX<T> segments_x;
    Eigen::VectorX<T> segments_y;
    Eigen::VectorX<T> gauss_weights_x;
    Eigen::VectorX<T> gauss_points_x;
    Eigen::VectorX<T> gauss_weights_y;
    Eigen::VectorX<T> gauss_points_y;

    SamplingSVE(AbstractKernel* K, double eps, int n_gauss_ = -1)
        : kernel(K), epsilon(eps), n_gauss(n_gauss_) {
        // Initialize based on K and epsilon
        nsvals_hint = sve_hints(kernel, epsilon);

        if (n_gauss == -1) {
            n_gauss = nsvals_hint;
        }

        // Initialize quadrature rules and segments
        // Placeholder code; actual implementation will depend on your quadrature rules

        // For demonstration, initialize with dummy data
        int num_segments = 1; // Placeholder
        segments_x = Eigen::VectorX<T>::LinSpaced(num_segments + 1, -1.0, 1.0);
        segments_y = Eigen::VectorX<T>::LinSpaced(num_segments + 1, -1.0, 1.0);

        gauss_points_x = Eigen::VectorX<T>::LinSpaced(n_gauss, -1.0, 1.0);
        gauss_weights_x = Eigen::VectorX<T>::Ones(n_gauss) / n_gauss;
        gauss_points_y = Eigen::VectorX<T>::LinSpaced(n_gauss, -1.0, 1.0);
        gauss_weights_y = Eigen::VectorX<T>::Ones(n_gauss) / n_gauss;
    }

    virtual std::vector<Eigen::MatrixX<T>> matrices() const {
        // Compute the matrices based on kernel and quadrature points/weights
        // Placeholder implementation

        Eigen::MatrixX<T> result(gauss_points_x.size(), gauss_points_y.size());

        for (int i = 0; i < gauss_points_x.size(); ++i) {
            for (int j = 0; j < gauss_points_y.size(); ++j) {
                // Compute K(x_i, y_j)
                // Placeholder: you need to replace this with actual kernel evaluation code
                double K_xy = 1.0; // Placeholder

                result(i, j) = std::sqrt(gauss_weights_x(i)) * K_xy * std::sqrt(gauss_weights_y(j));
            }
        }

        std::vector<Eigen::MatrixX<T>> mats;
        mats.push_back(result);
        return mats;
    }

    virtual SVEResult<T> postprocess(const std::vector<MatrixX<T>>& u_list,
                                  const std::vector<VectorX<T>>& s_list,
                                  const std::vector<MatrixX<T>>& v_list) {
        // Assume only one matrix in u_list, s_list, v_list
        const MatrixX<T>& u = u_list[0];
        const VectorX<T>& s = s_list[0];
        const MatrixX<T>& v = v_list[0];

        // Postprocess to construct the singular functions

        // Divide u and v by sqrt of weights
        Eigen::MatrixX<T> u_x = u.array().colwise() / gauss_weights_x.array().sqrt();
        Eigen::MatrixX<T> v_y = v.array().colwise() / gauss_weights_y.array().sqrt();

        // Reshape u_x and v_y into tensors (n_gauss, num_segments, s.size())
        // For simplicity, assume one segment
        int num_segments_x = segments_x.size() - 1;
        int num_segments_y = segments_y.size() - 1;

        // Placeholder: store u_x and v_y as is
        std::vector<Eigen::MatrixX<T>> u_data;
        std::vector<Eigen::MatrixX<T>> v_data;

        u_data.push_back(u_x);
        v_data.push_back(v_y);

        // Multiply by sqrt of segment lengths
        double dsegs_x = segments_x(1) - segments_x(0);
        double dsegs_y = segments_y(1) - segments_y(0);
        u_data[0] *= std::sqrt(0.5 * dsegs_x);
        v_data[0] *= std::sqrt(0.5 * dsegs_y);

        // Construct polynomials (or whatever representation is appropriate)
        // Here, we'll simply store u_data and v_data as u and v in SVEResult

        std::vector<Eigen::MatrixX<T>> u_funcs = u_data;
        std::vector<Eigen::MatrixX<T>> v_funcs = v_data;

        return SVEResult<T>(u_funcs, s, v_funcs, kernel, epsilon);
    }
};

// CentrosymmSVE class
template <typename T>
class CentrosymmSVE : public AbstractSVE<T> {
public:
    AbstractKernel* kernel;
    double epsilon;
    SamplingSVE<T>* even;
    SamplingSVE<T>* odd;
    int nsvals_hint;

    CentrosymmSVE(AbstractKernel* K, double eps, int n_gauss = -1)
        : kernel(K), epsilon(eps) {
        even = new SamplingSVE<T>(kernel->get_symmetrized(+1), epsilon, n_gauss);
        odd = new SamplingSVE<T>(kernel->get_symmetrized(-1), epsilon, n_gauss);

        nsvals_hint = std::max(even->nsvals_hint, odd->nsvals_hint);
    }

    ~CentrosymmSVE() {
        delete even;
        delete odd;
    }

    virtual std::vector<Eigen::MatrixX<T>> matrices() const {
        std::vector<Eigen::MatrixX<T>> matrices_even = even->matrices();
        std::vector<Eigen::MatrixX<T>> matrices_odd = odd->matrices();
        std::vector<Eigen::MatrixX<T>> result;
        result.insert(result.end(), matrices_even.begin(), matrices_even.end());
        result.insert(result.end(), matrices_odd.begin(), matrices_odd.end());
        return result;
    }

    virtual SVEResult<T> postprocess(const std::vector<Eigen::MatrixX<T>>& u_list,
                                  const std::vector<VectorX<T>>& s_list,
                                  const std::vector<Eigen::MatrixX<T>>& v_list) {
        // Postprocess even and odd components

        // Assuming u_list, s_list, v_list contain the outputs from both even and odd SVDs

        // Split even and odd components
        size_t half = u_list.size() / 2;

        std::vector<Eigen::MatrixX<T>> u_even(u_list.begin(), u_list.begin() + half);
        std::vector<VectorX<T>> s_even(s_list.begin(), s_list.begin() + half);
        std::vector<Eigen::MatrixX<T>> v_even(v_list.begin(), v_list.begin() + half);

        std::vector<Eigen::MatrixX<T>> u_odd(u_list.begin() + half, u_list.end());
        std::vector<VectorX<T>> s_odd(s_list.begin() + half, s_list.end());
        std::vector<Eigen::MatrixX<T>> v_odd(v_list.begin() + half, v_list.end());

        // Postprocess even and odd parts
        SVEResult<T> res_even = even->postprocess(u_even, s_even, v_even);
        SVEResult<T> res_odd = odd->postprocess(u_odd, s_odd, v_odd);

        // Merge the results
        std::vector<Eigen::MatrixX<T>> u_all = res_even.u;
        u_all.insert(u_all.end(), res_odd.u.begin(), res_odd.u.end());

        Eigen::VectorX<T> s_all(res_even.s.size() + res_odd.s.size());
        s_all << res_even.s, res_odd.s;

        std::vector<Eigen::MatrixX<T>> v_all = res_even.v;
        v_all.insert(v_all.end(), res_odd.v.begin(), res_odd.v.end());

        // Assign signs
        std::vector<int> signs(res_even.s.size() + res_odd.s.size());
        std::fill(signs.begin(), signs.begin() + res_even.s.size(), +1);
        std::fill(signs.begin() + res_even.s.size(), signs.end(), -1);

        // Sort singular values in descending order
        std::vector<size_t> indices(s_all.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&s_all](size_t i1, size_t i2) {
            return s_all(i1) > s_all(i2);
        });

        // Apply sorting to u, s, v, signs
        std::vector<Eigen::MatrixX<T>> u_sorted(s_all.size());
        Eigen::VectorX<T> s_sorted(s_all.size());
        std::vector<Eigen::MatrixX<T>> v_sorted(s_all.size());
        std::vector<int> signs_sorted(s_all.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            u_sorted[i] = u_all[indices[i]];
            s_sorted(i) = s_all(indices[i]);
            v_sorted[i] = v_all[indices[i]];
            signs_sorted[i] = signs[indices[i]];
        }

        // Extend to negative side (requires specific operations)
        // Placeholder: returning the sorted results
        return SVEResult<T>(u_sorted, s_sorted, v_sorted, kernel, epsilon);
    }
};

// Helper function to compute safe epsilon and work data type
inline std::tuple<double, int, std::string>
choose_accuracy(double epsilon, int work_dtype, std::string svd_strategy) {
    double safe_epsilon;
    int Twork = work_dtype;
    std::string auto_svd_strategy = svd_strategy;

    if (epsilon == 0 && Twork == 0) {
        safe_epsilon = sqrt_eps<double>();
        Twork = 1; // Assume 1 represents double
        auto_svd_strategy = "default";
    } else if (epsilon >= sqrt_eps<double>()) {
        safe_epsilon = epsilon;
        Twork = 1;
        auto_svd_strategy = "default";
    } else {
        // Handle cases where higher precision data type is needed
        safe_epsilon = epsilon;
        Twork = 2; // Assume 2 represents higher precision, e.g., double-double
        auto_svd_strategy = "default";
    }

    return std::make_tuple(safe_epsilon, Twork, auto_svd_strategy);
}

// Helper function to truncate singular values
template <typename T>
inline std::tuple<std::vector<Eigen::MatrixX<T>>, std::vector<VectorX<T>>, std::vector<Eigen::MatrixX<T>>>
truncate(const std::vector<Eigen::MatrixX<T>>& u_list,
         const std::vector<VectorX<T>>& s_list,
         const std::vector<Eigen::MatrixX<T>>& v_list,
         double rtol = 0.0,
         int lmax = std::numeric_limits<int>::max()) {
    int total_svs = 0;
    for (const auto& s : s_list) {
        total_svs += s.size();
    }

    std::vector<double> all_singular_values;
    for (const auto& s : s_list) {
        for (int i = 0; i < s.size(); ++i) {
            all_singular_values.push_back(s(i));
        }
    }

    std::sort(all_singular_values.begin(), all_singular_values.end(), std::greater<double>());
    double cutoff = rtol * all_singular_values[0];

    if (lmax < total_svs) {
        cutoff = std::max(cutoff, all_singular_values[lmax - 1]);
    }

    std::vector<Eigen::MatrixX<T>> u_cut;
    std::vector<VectorX<T>> s_cut;
    std::vector<Eigen::MatrixX<T>> v_cut;

    for (size_t idx = 0; idx < u_list.size(); ++idx) {
        const Eigen::MatrixX<T>& u = u_list[idx];
        const Eigen::VectorX<T>& s = s_list[idx];
        const Eigen::MatrixX<T>& v = v_list[idx];

        int scount = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (s(i) > cutoff) {
                ++scount;
            }
        }
        if (scount > 0) {
            u_cut.push_back(u.leftCols(scount));
            s_cut.push_back(s.head(scount));
            v_cut.push_back(v.leftCols(scount));
        }
    }

    return std::make_tuple(u_cut, s_cut, v_cut);
}

// Canonicalize function
template <typename T>
inline void canonicalize(std::vector<Eigen::MatrixX<T>>& u_list,
                  std::vector<Eigen::MatrixX<T>>& v_list) {
    for (size_t i = 0; i < u_list.size(); ++i) {
        double gauge = sign(u_list[i](0, 0)); // Evaluate u[i] at 1; placeholder
        u_list[i] *= gauge;
        v_list[i] *= gauge;
    }
}

// Main compute function
template <typename T>
inline SVEResult<T> compute(AbstractKernel* kernel,
                  double epsilon = 0,
                  double cutoff = -1,
                  int lmax = std::numeric_limits<int>::max(),
                  int n_gauss = -1,
                  int work_dtype = 0,
                  std::string svd_strategy = "auto",
                  AbstractSVE<T>* SVE_strategy = nullptr) {
    double safe_epsilon;
    int Twork;
    std::string svd_strat;

    std::tie(safe_epsilon, Twork, svd_strat) = choose_accuracy(epsilon, work_dtype, svd_strategy);
    if (svd_strategy != "auto") {
        svd_strat = svd_strategy;
    }

    // Create SVE strategy
    AbstractSVE<T>* sve;
    if (SVE_strategy != nullptr) {
        sve = SVE_strategy;
    } else {
        if (kernel->is_centrosymmetric()) {
            sve = new CentrosymmSVE<T>(kernel, safe_epsilon, n_gauss);
        } else {
            sve = new SamplingSVE<T>(kernel, safe_epsilon, n_gauss);
        }
    }

    std::vector<Eigen::MatrixX<T>> matrices = sve->matrices();

    std::vector<Eigen::MatrixX<T>> u_list;
    std::vector<VectorX<T>> s_list;
    std::vector<Eigen::MatrixX<T>> v_list;

    // Compute SVDs of the matrices
    for (const auto& mat : matrices) {
        Eigen::JacobiSVD<Eigen::MatrixX<T>> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixX<T> u = svd.matrixU();
        Eigen::VectorX<T> s = svd.singularValues();
        Eigen::MatrixX<T> v = svd.matrixV();

        u_list.push_back(u);
        s_list.push_back(s);
        v_list.push_back(v);
    }

    if (cutoff == -1) {
        cutoff = 2 * std::numeric_limits<double>::epsilon();
    }

    // Truncate singular values
    std::tie(u_list, s_list, v_list) = truncate(u_list, s_list, v_list, cutoff, lmax);

    // Postprocess
    SVEResult<T> result = sve->postprocess(u_list, s_list, v_list);

    // Canonicalize
    canonicalize(result.u, result.v);

    delete sve;
    return result;
}

} // namespace sparseir