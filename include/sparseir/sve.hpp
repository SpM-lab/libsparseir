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

// Forward declarations
class AbstractSVE;
class SamplingSVE;
class CentrosymmSVE;
class SVEResult;
class AbstractKernel;

// Type aliases
typedef double Float64;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;

// Function templates
template<typename T>
inline T sqrt_eps() {
    return std::sqrt(std::numeric_limits<T>::epsilon());
}

// Helper function for sign
inline double sign(double x) {
    return (x > 0) - (x < 0);
}

// AbstractKernel class
class AbstractKernel {
public:
    virtual ~AbstractKernel() {}

    virtual bool is_centrosymmetric() const = 0;
    virtual AbstractKernel* get_symmetrized(int sym_type) const = 0; // sym_type +1 or -1
    virtual int sve_hints(double epsilon) const = 0;
    // Other methods and members...
};

// AbstractSVE class
class AbstractSVE {
public:
    virtual ~AbstractSVE() {}
    virtual std::vector<Matrix> matrices() const = 0;
    virtual SVEResult postprocess(const std::vector<Matrix>& u_list,
                                  const std::vector<Vector>& s_list,
                                  const std::vector<Matrix>& v_list) = 0;
    int nsvals_hint;
};

// SVEResult class
class SVEResult {
public:
    // Assume PiecewiseLegendrePolyVector is a user-defined type representing the polynomials
    // For this example, we'll use vectors of Eigen matrices as placeholders
    std::vector<Eigen::MatrixXd> u; // Left singular functions
    Vector s; // Singular values
    std::vector<Eigen::MatrixXd> v; // Right singular functions

    AbstractKernel* kernel;
    double epsilon;

    SVEResult(const std::vector<Eigen::MatrixXd>& u_,
              const Vector& s_,
              const std::vector<Eigen::MatrixXd>& v_,
              AbstractKernel* K,
              double eps)
        : u(u_), s(s_), v(v_), kernel(K), epsilon(eps) {}

    std::tuple<std::vector<Eigen::MatrixXd>, Vector, std::vector<Eigen::MatrixXd>>
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
            std::vector<Eigen::MatrixXd> u_cut(u.begin(), u.begin() + cut);
            Vector s_cut = s.head(cut);
            std::vector<Eigen::MatrixXd> v_cut(v.begin(), v.begin() + cut);
            return std::make_tuple(u_cut, s_cut, v_cut);
        }
    }
};

// SamplingSVE class
class SamplingSVE : public AbstractSVE {
public:
    AbstractKernel* kernel;
    double epsilon;
    int n_gauss;
    int nsvals_hint;

    // Additional members to store quadrature rules, segments, etc.
    // For simplicity, we'll use Eigen types to represent them
    Eigen::VectorXd segments_x;
    Eigen::VectorXd segments_y;
    Eigen::VectorXd gauss_weights_x;
    Eigen::VectorXd gauss_points_x;
    Eigen::VectorXd gauss_weights_y;
    Eigen::VectorXd gauss_points_y;

    SamplingSVE(AbstractKernel* K, double eps, int n_gauss_ = -1)
        : kernel(K), epsilon(eps), n_gauss(n_gauss_) {
        // Initialize based on K and epsilon
        nsvals_hint = kernel->sve_hints(epsilon);

        if (n_gauss == -1) {
            n_gauss = nsvals_hint;
        }

        // Initialize quadrature rules and segments
        // Placeholder code; actual implementation will depend on your quadrature rules

        // For demonstration, initialize with dummy data
        int num_segments = 1; // Placeholder
        segments_x = Eigen::VectorXd::LinSpaced(num_segments + 1, -1.0, 1.0);
        segments_y = Eigen::VectorXd::LinSpaced(num_segments + 1, -1.0, 1.0);

        gauss_points_x = Eigen::VectorXd::LinSpaced(n_gauss, -1.0, 1.0);
        gauss_weights_x = Eigen::VectorXd::Ones(n_gauss) / n_gauss;
        gauss_points_y = Eigen::VectorXd::LinSpaced(n_gauss, -1.0, 1.0);
        gauss_weights_y = Eigen::VectorXd::Ones(n_gauss) / n_gauss;
    }

    virtual std::vector<Matrix> matrices() const {
        // Compute the matrices based on kernel and quadrature points/weights
        // Placeholder implementation

        Matrix result(gauss_points_x.size(), gauss_points_y.size());

        for (int i = 0; i < gauss_points_x.size(); ++i) {
            for (int j = 0; j < gauss_points_y.size(); ++j) {
                // Compute K(x_i, y_j)
                // Placeholder: you need to replace this with actual kernel evaluation code
                double K_xy = 1.0; // Placeholder

                result(i, j) = std::sqrt(gauss_weights_x(i)) * K_xy * std::sqrt(gauss_weights_y(j));
            }
        }

        std::vector<Matrix> mats;
        mats.push_back(result);
        return mats;
    }

    virtual SVEResult postprocess(const std::vector<Matrix>& u_list,
                                  const std::vector<Vector>& s_list,
                                  const std::vector<Matrix>& v_list) {
        // Assume only one matrix in u_list, s_list, v_list
        const Matrix& u = u_list[0];
        const Vector& s = s_list[0];
        const Matrix& v = v_list[0];

        // Postprocess to construct the singular functions

        // Divide u and v by sqrt of weights
        Matrix u_x = u.array().colwise() / gauss_weights_x.array().sqrt();
        Matrix v_y = v.array().colwise() / gauss_weights_y.array().sqrt();

        // Reshape u_x and v_y into tensors (n_gauss, num_segments, s.size())
        // For simplicity, assume one segment
        int num_segments_x = segments_x.size() - 1;
        int num_segments_y = segments_y.size() - 1;

        // Placeholder: store u_x and v_y as is
        std::vector<Eigen::MatrixXd> u_data;
        std::vector<Eigen::MatrixXd> v_data;

        u_data.push_back(u_x);
        v_data.push_back(v_y);

        // Multiply by sqrt of segment lengths
        double dsegs_x = segments_x(1) - segments_x(0);
        double dsegs_y = segments_y(1) - segments_y(0);
        u_data[0] *= std::sqrt(0.5 * dsegs_x);
        v_data[0] *= std::sqrt(0.5 * dsegs_y);

        // Construct polynomials (or whatever representation is appropriate)
        // Here, we'll simply store u_data and v_data as u and v in SVEResult

        std::vector<Eigen::MatrixXd> u_funcs = u_data;
        std::vector<Eigen::MatrixXd> v_funcs = v_data;

        return SVEResult(u_funcs, s, v_funcs, kernel, epsilon);
    }
};

// CentrosymmSVE class
class CentrosymmSVE : public AbstractSVE {
public:
    AbstractKernel* kernel;
    double epsilon;
    SamplingSVE* even;
    SamplingSVE* odd;
    int nsvals_hint;

    CentrosymmSVE(AbstractKernel* K, double eps, int n_gauss = -1)
        : kernel(K), epsilon(eps) {
        even = new SamplingSVE(kernel->get_symmetrized(+1), epsilon, n_gauss);
        odd = new SamplingSVE(kernel->get_symmetrized(-1), epsilon, n_gauss);

        nsvals_hint = std::max(even->nsvals_hint, odd->nsvals_hint);
    }

    ~CentrosymmSVE() {
        delete even;
        delete odd;
    }

    virtual std::vector<Matrix> matrices() const {
        std::vector<Matrix> matrices_even = even->matrices();
        std::vector<Matrix> matrices_odd = odd->matrices();
        std::vector<Matrix> result;
        result.insert(result.end(), matrices_even.begin(), matrices_even.end());
        result.insert(result.end(), matrices_odd.begin(), matrices_odd.end());
        return result;
    }

    virtual SVEResult postprocess(const std::vector<Matrix>& u_list,
                                  const std::vector<Vector>& s_list,
                                  const std::vector<Matrix>& v_list) {
        // Postprocess even and odd components

        // Assuming u_list, s_list, v_list contain the outputs from both even and odd SVDs

        // Split even and odd components
        size_t half = u_list.size() / 2;

        std::vector<Matrix> u_even(u_list.begin(), u_list.begin() + half);
        std::vector<Vector> s_even(s_list.begin(), s_list.begin() + half);
        std::vector<Matrix> v_even(v_list.begin(), v_list.begin() + half);

        std::vector<Matrix> u_odd(u_list.begin() + half, u_list.end());
        std::vector<Vector> s_odd(s_list.begin() + half, s_list.end());
        std::vector<Matrix> v_odd(v_list.begin() + half, v_list.end());

        // Postprocess even and odd parts
        SVEResult res_even = even->postprocess(u_even, s_even, v_even);
        SVEResult res_odd = odd->postprocess(u_odd, s_odd, v_odd);

        // Merge the results
        std::vector<Eigen::MatrixXd> u_all = res_even.u;
        u_all.insert(u_all.end(), res_odd.u.begin(), res_odd.u.end());

        Vector s_all(res_even.s.size() + res_odd.s.size());
        s_all << res_even.s, res_odd.s;

        std::vector<Eigen::MatrixXd> v_all = res_even.v;
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
        std::vector<Eigen::MatrixXd> u_sorted(s_all.size());
        Vector s_sorted(s_all.size());
        std::vector<Eigen::MatrixXd> v_sorted(s_all.size());
        std::vector<int> signs_sorted(s_all.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            u_sorted[i] = u_all[indices[i]];
            s_sorted(i) = s_all(indices[i]);
            v_sorted[i] = v_all[indices[i]];
            signs_sorted[i] = signs[indices[i]];
        }

        // Extend to negative side (requires specific operations)
        // Placeholder: returning the sorted results
        return SVEResult(u_sorted, s_sorted, v_sorted, kernel, epsilon);
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
inline std::tuple<std::vector<Matrix>, std::vector<Vector>, std::vector<Matrix>>
truncate(const std::vector<Matrix>& u_list,
         const std::vector<Vector>& s_list,
         const std::vector<Matrix>& v_list,
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

    std::vector<Matrix> u_cut;
    std::vector<Vector> s_cut;
    std::vector<Matrix> v_cut;

    for (size_t idx = 0; idx < u_list.size(); ++idx) {
        const Matrix& u = u_list[idx];
        const Vector& s = s_list[idx];
        const Matrix& v = v_list[idx];

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
inline void canonicalize(std::vector<Eigen::MatrixXd>& u_list,
                  std::vector<Eigen::MatrixXd>& v_list) {
    for (size_t i = 0; i < u_list.size(); ++i) {
        double gauge = sign(u_list[i](0, 0)); // Evaluate u[i] at 1; placeholder
        u_list[i] *= gauge;
        v_list[i] *= gauge;
    }
}

// Main compute function
inline SVEResult compute(AbstractKernel* kernel,
                  double epsilon = 0,
                  double cutoff = -1,
                  int lmax = std::numeric_limits<int>::max(),
                  int n_gauss = -1,
                  int work_dtype = 0,
                  std::string svd_strategy = "auto",
                  AbstractSVE* SVE_strategy = nullptr) {
    double safe_epsilon;
    int Twork;
    std::string svd_strat;

    std::tie(safe_epsilon, Twork, svd_strat) = choose_accuracy(epsilon, work_dtype, svd_strategy);
    if (svd_strategy != "auto") {
        svd_strat = svd_strategy;
    }

    // Create SVE strategy
    AbstractSVE* sve;
    if (SVE_strategy != nullptr) {
        sve = SVE_strategy;
    } else {
        if (kernel->is_centrosymmetric()) {
            sve = new CentrosymmSVE(kernel, safe_epsilon, n_gauss);
        } else {
            sve = new SamplingSVE(kernel, safe_epsilon, n_gauss);
        }
    }

    std::vector<Matrix> matrices = sve->matrices();

    std::vector<Matrix> u_list;
    std::vector<Vector> s_list;
    std::vector<Matrix> v_list;

    // Compute SVDs of the matrices
    for (const auto& mat : matrices) {
        Eigen::JacobiSVD<Matrix> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Matrix u = svd.matrixU();
        Vector s = svd.singularValues();
        Matrix v = svd.matrixV();

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
    SVEResult result = sve->postprocess(u_list, s_list, v_list);

    // Canonicalize
    canonicalize(result.u, result.v);

    delete sve;
    return result;
}

} // namespace sparseir