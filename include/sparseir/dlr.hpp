#pragma once

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <vector>

namespace sparseir {

template <typename Statistics>
class MatsubaraPoles {
public:
    double beta;
    Eigen::VectorXd poles;
    double wmax;
    std::vector<double> weights;

    MatsubaraPoles(double beta, const Eigen::VectorXd &poles, double wmax, std::function<double(double, double)> weight_func)
        : beta(beta), poles(poles), wmax(wmax), weights(poles.size(), 1.0)
    {
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            weights[i] = weight_func(beta, poles(i));
        }
    }

    template <typename T = Statistics>
    MatsubaraPoles(double beta, const Eigen::VectorXd &poles, double wmax,
                   typename std::enable_if<std::is_same<T, Fermionic>::value>::type* = nullptr)
        : beta(beta), poles(poles), wmax(wmax), weights(poles.size(), 1.0)
    {
    }


    // Return the size of the poles vector
    std::size_t size() const { return poles.size(); }

    // For Fermionic case
    template <typename T = Statistics>
    typename std::enable_if<std::is_same<T, Fermionic>::value,
                            Eigen::VectorXcd>::type
    operator()(const MatsubaraFreq<Fermionic> &n) const
    {
        Eigen::VectorXcd result(poles.size());
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            result(i) = 1.0 / (n.valueim(beta) - poles(i));
        }
        return result;
    }

    // For Bosonic case
    template <typename T = Statistics>
    typename std::enable_if<std::is_same<T, Bosonic>::value,
                            Eigen::VectorXcd>::type
    operator()(const MatsubaraFreq<Bosonic> &n) const
    {
        Eigen::VectorXcd result(poles.size());
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            double y = poles(i) / wmax;
            double weight = weights[i];
            result(i) = 1.0 / ((n.valueim(beta) - poles(i)) * weight);
        }
        return result;
    }


    // For vector of frequencies
    template <typename FreqType>
    Eigen::MatrixXcd operator()(const std::vector<FreqType> &n) const
    {
        Eigen::MatrixXcd result(poles.size(), n.size());
        for (size_t i = 0; i < n.size(); ++i) {
            result.col(i) = (*this)(MatsubaraFreq<Statistics>(n[i]));
        }
        return result;
    }

    Eigen::MatrixXcd operator()(const Eigen::Array<int64_t, Eigen::Dynamic, 1> &n_array) const {
        // delegate to operator()(const std::vector<FreqType> &n)
        return (*this)(std::vector<MatsubaraFreq<Statistics>>(n_array.data(), n_array.data() + n_array.size()));
    }
};


template <typename S>
class DLRBasisFunctions {
public:
    double beta;
    Eigen::VectorXd poles;
    double wmax;
    Eigen::VectorXd weights;

    DLRBasisFunctions(double beta, const Eigen::VectorXd &poles, double wmax, const Eigen::VectorXd &weights)
        : beta(beta), poles(poles), wmax(wmax), weights(weights)
    {
    }

    template <typename T = S>
    DLRBasisFunctions(double beta, const Eigen::VectorXd &poles, double wmax, typename std::enable_if<std::is_same<T, Fermionic>::value>::type* = nullptr)
        : beta(beta), poles(poles), wmax(wmax), weights(Eigen::VectorXd::Ones(poles.size()))
    {
    }

    // Evaluate at tau points
    template <typename T = S,
              typename std::enable_if<std::is_same<T, Fermionic>::value, int>::type = 0>
    Eigen::VectorXd operator()(double tau) const
    {
        Eigen::VectorXd result(poles.size());
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            auto kernel = LogisticKernel(beta * wmax); // k(x, y)
            double x = 2 * tau / beta - 1.0;
            double y = poles(i) / wmax;
            result(i) = - kernel.compute(x, y) / weights(i);
        }
        return result;
    }

    template <typename T = S,
              typename std::enable_if<std::is_same<T, Bosonic>::value, int>::type = 0>
    Eigen::VectorXd operator()(double tau) const
    {
        // k(x, y) = y * exp(-Λ y (x + 1) / 2) / (1 - exp(-Λ y))
        auto kernel = RegularizedBoseKernel(beta * wmax); // k(x, y)
        Eigen::VectorXd result(poles.size());
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            double x = 2.0 * tau / beta - 1.0;
            double y = poles(i) / wmax;
            double xtau = poles(i) * tau;
            double k_tau_omega = kernel.compute(x, y);
            result(i) = - k_tau_omega / (y * weights(i));
        }
        return result;
    }

    // For vector of tau points
    Eigen::MatrixXd operator()(const Eigen::VectorXd &tau) const
    {
        Eigen::MatrixXd result(poles.size(), tau.size());
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            result.row(i) = (*this)(tau);
        }
        return result;
    }

    std::size_t size() const { return poles.size(); }

    DLRBasisFunctions slice(size_t i) const {
        return DLRBasisFunctions(beta, poles.segment(i, 1), wmax, weights.segment(i, 1));
    }
};


template <typename S>
Eigen::VectorXd default_omega_sampling_points(const FiniteTempBasis<S> &basis)
{
    Eigen::VectorXd y = default_sampling_points(*(basis.sve_result->v), basis.size());

    // If the number of points is even, return as is
    if (y.size() % 2 == 0) {
        return basis.get_wmax() * y;
    }

    // For odd number of points, we need to handle the symmetry
    int n = y.size();
    int n_half = (n - 1) / 2;  // number of positive frequencies excluding zero

    // Create a new vector with 2*n_half + 2 elements
    Eigen::VectorXd y_new(2*n_half + 2);

    // Copy positive frequencies (excluding zero)
    y_new.head(n_half) = y.tail(n_half);

    // Add the new points (half of the smallest positive frequency)
    double min_pos = y_new(0);  // smallest positive frequency
    y_new(n_half) = min_pos/2.0;
    y_new(n_half+1) = -min_pos/2.0;

    // Copy positive frequencies to negative side
    y_new.tail(n_half) = -y_new.head(n_half);

    // Sort the frequencies
    std::sort(y_new.data(), y_new.data() + y_new.size());

    return basis.get_wmax() * y_new;
}


template<typename S>
using DLRTauFuncsType = PeriodicFunctions<S, DLRBasisFunctions<S>>;

template <typename S>
class DiscreteLehmannRepresentation : public AbstractBasis<S> {
public:
    Eigen::VectorXd poles;
    double beta;
    double wmax;
    double accuracy;
    std::shared_ptr<DLRTauFuncsType<S>> u;
    std::shared_ptr<MatsubaraPoles<S>> uhat;
    Eigen::MatrixXd fitmat;
    Eigen::JacobiSVD<Eigen::MatrixXd> matrix;
    std::function<double(double, double)> weight_func;
    Eigen::VectorXd _ir_default_tau_sampling_points;


    // Constructor with basis and poles
    DiscreteLehmannRepresentation(const FiniteTempBasis<S> &b,
                                  const Eigen::VectorXd &poles)
        : beta(b.get_beta()), wmax(b.get_wmax()), poles(poles), accuracy(b.get_accuracy()),
          u(nullptr),
          uhat(std::make_shared<MatsubaraPoles<S>>(b.get_beta(), poles, b.get_wmax(), b.weight_func)),
          _ir_default_tau_sampling_points(b.default_tau_sampling_points())
    {
        // initialize u
        /*
        std::vector<std::shared_ptr<TauFunction<S, DLRBasisFunction<S>>>> u_funcs;
        for (int i = 0; i < poles.size(); ++i) {
            u_funcs.push_back(
                std::make_shared<TauFunction<S, DLRBasisFunction<S>>>(
                    std::make_shared<DLRBasisFunction<S>>(
                        b.get_beta(), poles[i], b.get_wmax(), b.weight_func(beta, poles[i])
                    ),
                    b.get_beta()
                )
            );
        }
        */

        Eigen::VectorXd weights(poles.size());
        for (int i = 0; i < poles.size(); ++i) {
            weights(i) = b.weight_func(beta, poles[i]);
        }
        auto base_u_funcs = std::make_shared<DLRBasisFunctions<S>>(b.get_beta(), poles, b.get_wmax(), weights);
        this->u = std::make_shared<DLRTauFuncsType<S>>(base_u_funcs, b.get_beta());

        // Fitting matrix from IR
        Eigen::MatrixXd A = (*b.v)(poles);
        Eigen::ArrayXXd A_array = A.array();
        Eigen::ArrayXd s_array = b.s.array();

        // Perform element-wise multiplication
        // size: (size of basis, size of poles)
        fitmat = (-A_array * s_array.replicate(1, A.cols())).matrix();

        matrix.compute(fitmat, Eigen::ComputeThinU | Eigen::ComputeThinV);

        weight_func = b.weight_func;
        wmax = b.get_wmax();
    }

    // Constructor with just basis
    explicit DiscreteLehmannRepresentation(const FiniteTempBasis<S> &b)
        : DiscreteLehmannRepresentation(b, default_omega_sampling_points(b))
    {
    }

    // Required virtual function implementations
    size_t size() const override { return poles.size(); }
    const Eigen::VectorXd significance() const override
    {
        return Eigen::VectorXd::Ones(size());
    }
    double get_accuracy() const override { return accuracy; }
    Eigen::VectorXd default_tau_sampling_points() const override
    {
        throw std::runtime_error("default_tau_sampling_points is not implemented for DiscreteLehmannRepresentation");
        // return length 0 vector
        return Eigen::VectorXd::Zero(0);
    }

    double get_wmax() const override { return wmax; }
    double get_beta() const override { return beta; }

    std::vector<MatsubaraFreq<S>>
    default_matsubara_sampling_points(int L, bool fence = false,
                                      bool positive_only = false) const override
    {
        (void)L; // Silence unused parameter warning
        (void)fence; // Silence unused parameter warning
        (void)positive_only; // Silence unused parameter warning
        throw std::runtime_error("default_matsubara_sampling_points is not implemented for DiscreteLehmannRepresentation");
        return std::vector<MatsubaraFreq<S>>();
        //return basis.default_matsubara_sampling_points(L, fence, positive_only);
    }

    template <typename T, int N>
    Eigen::Tensor<T, N> from_IR(const Eigen::Tensor<T, N> &ir)
    {
        // Let fitmat have shape (r, c)
        const int r = fitmat.rows();
        const int c = fitmat.cols();

        // Get dimensions of input tensor 'ir' (should be (r, d2, d3, ...))
        std::array<Eigen::Index, N> ir_dims;
        for (int i = 0; i < N; ++i) {
            ir_dims[i] = ir.dimension(i);
        }
        // Ensure that the first dimension of ir matches the rows of fitmat
        assert(ir_dims[0] == r && "Mismatch between fitmat.rows() and first "
                                  "dimension of input tensor ir");

        // Compute total number of slices = product of dimensions from index 1
        // to N-1
        Eigen::Index numSlices = 1;
        for (int i = 1; i < N; ++i) {
            numSlices *= ir_dims[i];
        }

        // Map 'ir' to a matrix of shape (r, numSlices).
        // This works if the tensor's data is stored contiguously.
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
            ir_mat(ir.data(), r, numSlices);

        // Prepare a matrix to hold the solution X, which will have shape (c,
        // numSlices)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X(c, numSlices);

        // Solve the linear systems: fitmat * X = ir_mat.
        // Here we use ColPivHouseholderQR (you could also use another solver if
        // desired).
        Eigen::ColPivHouseholderQR<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
            solver(fitmat);
        X = solver.solve(ir_mat);

        // Define output tensor dimensions: first dimension becomes c, and the
        // remaining dimensions are unchanged.
        std::array<Eigen::Index, N> out_dims;
        out_dims[0] = c;
        for (int i = 1; i < N; ++i) {
            out_dims[i] = ir_dims[i];
        }

        // Create an output tensor and copy the data from X into it.
        Eigen::Tensor<T, N> result(out_dims);
        std::copy(X.data(), X.data() + X.size(), result.data());

        return result;
    }

    // Convert from DLR to IR
    template <typename T, int N>
    Eigen::Tensor<T, N> to_IR(const Eigen::Tensor<T, N> &g_dlr, int target_dim=0) const
    {
        // fitmat is a matrix, so we need to convert it to a tensor
        Eigen::TensorMap<Eigen::Tensor<const double, 2>> fitmat_as_tensor(
            fitmat.data(), fitmat.rows(), fitmat.cols());
        std::array<Eigen::IndexPair<int>, 1> contraction_pairs = {
            Eigen::IndexPair<int>(1, target_dim)};
        return sparseir::movedim<T, N>(fitmat_as_tensor.contract(g_dlr, contraction_pairs), 0, target_dim);
    }

};

} // namespace sparseir