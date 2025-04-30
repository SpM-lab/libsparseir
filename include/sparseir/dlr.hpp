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

    Eigen::MatrixXcd operator()(const Eigen::ArrayXi &n_array) const {
        // delegate to operator()(const std::vector<FreqType> &n)
        return (*this)(std::vector<MatsubaraFreq<Statistics>>(n_array.data(), n_array.data() + n_array.size()));
    }
};

template <typename S>
class TauPoles {
public:
    double beta;
    Eigen::VectorXd poles;
    double wmax;
    std::vector<double> weights;

    TauPoles(double beta, const Eigen::VectorXd &poles, double wmax, std::function<double(double, double)> weight_func)
        : beta(beta), poles(poles), wmax(wmax), weights(poles.size(), 1.0)
    {
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            weights[i] = weight_func(beta, poles(i));
        }
    }

    template <typename T = S>
    TauPoles(double beta, const Eigen::VectorXd &poles, double wmax, typename std::enable_if<std::is_same<T, Fermionic>::value>::type* = nullptr)
        : beta(beta), poles(poles), wmax(wmax), weights(poles.size(), 1.0)
    {
    }


    // Return the size of the poles vector
    std::size_t size() const { return poles.size(); }

    // Evaluate at tau points
    template <typename T = S,
              typename std::enable_if<std::is_same<T, Fermionic>::value, int>::type = 0>
    Eigen::VectorXd operator()(double tau) const
    {
        Eigen::VectorXd result(poles.size());
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            double x = poles(i);
            double xtau = x * tau;
            result(i) = -std::exp(-xtau) / (1.0 + std::exp(-beta * x)) / weights[i];
        }
        return result;
    }

    template <typename T = S,
              typename std::enable_if<std::is_same<T, Bosonic>::value, int>::type = 0>
    Eigen::VectorXd operator()(double tau) const
    {
        Eigen::VectorXd result(poles.size());

        // k(x, y) = y * exp(-Λ y (x + 1) / 2) / (1 - exp(-Λ y))
        auto kernel = RegularizedBoseKernel(beta * wmax); // k(x, y)

        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            double w = poles(i);
            double x = 2.0 * tau / beta - 1.0;
            double y = w / wmax;
            double xtau = w * tau;

            // k(τ, ω) = ωmax * k(2τ/β - 1, ω/ωmax) = ω * exp(- τ * ω) / (1 - exp(-beta * ω))
            double k_tau_omega = wmax * kernel.compute(x, y);

            // FIXME: accuracy is not good for beta * x ~ 0.0
            //result(i) = -std::exp(-xtau) / (1.0 - std::exp(-beta * w)) / weights[i];
            //std::cout << "k_tau_omega: " << k_tau_omega << std::endl;
            //std::cout << "w: " << w << std::endl;
            //std::cout << "weights[i]: " << weights[i] << std::endl;
            result(i) = - k_tau_omega / (w * weights[i]);
        }
        return result;
    }

    // For vector of tau points
    Eigen::MatrixXd operator()(const Eigen::VectorXd &tau) const
    {
        Eigen::MatrixXd result(poles.size(), tau.size());
        for (Eigen::Index i = 0; i < tau.size(); ++i) {
            result.col(i) = (*this)(tau(i));
        }
        return result;
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

template <typename S>
class DiscreteLehmannRepresentation : public AbstractBasis<S> {
public:
    FiniteTempBasis<S> basis;
    Eigen::VectorXd poles;
    double wmax;
    std::shared_ptr<TauPoles<S>> u;
    std::shared_ptr<MatsubaraPoles<S>> uhat;
    Eigen::MatrixXd fitmat;
    Eigen::JacobiSVD<Eigen::MatrixXd> matrix;
    std::function<double(double, double)> weight_func;

    // Constructor with basis and poles
    DiscreteLehmannRepresentation(const FiniteTempBasis<S> &b,
                                  const Eigen::VectorXd &poles)
        : basis(b),
          poles(poles),
          u(std::make_shared<TauPoles<S>>(b.get_beta(), poles, basis.get_wmax(), basis.weight_func)),
          uhat(std::make_shared<MatsubaraPoles<S>>(b.get_beta(), poles, basis.get_wmax(), basis.weight_func))
    {
        // Fitting matrix from IR
        Eigen::MatrixXd A = (*basis.v)(poles);
        Eigen::ArrayXXd A_array = A.array();
        Eigen::ArrayXd s_array = basis.s.array();

        // Perform element-wise multiplication
        fitmat = (-A_array * s_array.replicate(1, A.cols())).matrix();

        matrix.compute(fitmat, Eigen::ComputeThinU | Eigen::ComputeThinV);

        weight_func = basis.weight_func;
        wmax = basis.get_wmax();
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
    double get_accuracy() const override { return basis.get_accuracy(); }
    double get_wmax() const override { return basis.get_wmax(); }
    Eigen::VectorXd default_tau_sampling_points() const override
    {
        return basis.default_tau_sampling_points();
    }

    std::vector<MatsubaraFreq<S>>
    default_matsubara_sampling_points(int L, bool fence = false,
                                      bool positive_only = false) const override
    {
        return basis.default_matsubara_sampling_points(L, fence, positive_only);
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
    Eigen::Tensor<T, N> to_IR(const Eigen::Tensor<T, N> &g_dlr) const
    {
        // fitmat is a matrix, so we need to convert it to a tensor
        Eigen::TensorMap<Eigen::Tensor<const double, 2>> fitmat_as_tensor(
            fitmat.data(), fitmat.rows(), fitmat.cols());
        std::array<Eigen::IndexPair<int>, 1> contraction_pairs = {
            Eigen::IndexPair<int>(1, 0)};
        return fitmat_as_tensor.contract(g_dlr, contraction_pairs);
    }

    double beta() const { return basis.beta; }
    double Lambda() const { return basis.Lambda(); }
};

} // namespace sparseir