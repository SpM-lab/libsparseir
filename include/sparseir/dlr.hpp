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

    MatsubaraPoles(double beta, const Eigen::VectorXd& poles)
        : beta(beta), poles(poles) { }

    // For Fermionic case
    template <typename T = Statistics>
    typename std::enable_if<std::is_same<T, Fermionic>::value, Eigen::VectorXcd>::type
    operator()(const MatsubaraFreq<Fermionic>& n) const {
        Eigen::VectorXcd result(poles.size());
        for(Eigen::Index i = 0; i < poles.size(); ++i) {
            result(i) = 1.0 / (valueim<Fermionic>(n, beta) - poles(i));
        }
        return result;
    }

    // For Bosonic case
    template <typename T = Statistics>
    typename std::enable_if<std::is_same<T, Bosonic>::value, Eigen::VectorXcd>::type
    operator()(const MatsubaraFreq<Bosonic>& n) const {
        Eigen::VectorXcd result(poles.size());
        for(Eigen::Index i = 0; i < poles.size(); ++i) {
            result(i) = std::tanh(beta * poles(i) / 2.0) /
                       (valueim<Bosonic>(n, beta) - poles(i));
        }
        return result;
    }

    // For vector of frequencies
    template <typename FreqType>
    Eigen::MatrixXcd operator()(const std::vector<FreqType>& n) const {
        Eigen::MatrixXcd result(poles.size(), n.size());
        for(size_t i = 0; i < n.size(); ++i) {
            result.col(i) = (*this)(MatsubaraFreq<Statistics>(n[i]));
        }
        return result;
    }
};

template <typename Statistics>
class TauPoles {
public:
    double beta;
    Eigen::VectorXd poles;
    double omega_max;

    TauPoles(double beta, const Eigen::VectorXd& poles)
        : beta(beta), poles(poles), omega_max(poles.array().abs().maxCoeff()) { }

    // Evaluate at tau points
    Eigen::VectorXd operator()(double tau) const {
        Eigen::VectorXd result(poles.size());
        for(Eigen::Index i = 0; i < poles.size(); ++i) {
            double x = poles(i);
            double xtau = x * tau;
            if (std::is_same<Statistics, Fermionic>::value) {
                result(i) = -std::exp(-xtau) / (1.0 + std::exp(-beta * x));
            } else {
                result(i) = std::exp(-xtau) / (1.0 - std::exp(-beta * x));
            }
        }
        return result;
    }

    // For vector of tau points
    Eigen::MatrixXd operator()(const Eigen::VectorXd& tau) const {
        Eigen::MatrixXd result(poles.size(), tau.size());
        for(Eigen::Index i = 0; i < tau.size(); ++i) {
            result.col(i) = (*this)(tau(i));
        }
        return result;
    }
};

template<typename Basis>
Eigen::VectorXd default_omega_sampling_points(const Basis& b) {
    double wmax = b.get_wmax();
    int n = b.size();
    Eigen::VectorXd omega(n);
    for (int i = 0; i < n; ++i) {
        omega(i) = wmax * std::cos(M_PI * (2 * i + 1) / (2 * n));
    }
    return omega;
}

template <typename Statistics, typename Basis>
class DiscreteLehmannRepresentation : public AbstractBasis<Statistics> {
public:
    Basis basis;
    Eigen::VectorXd poles;
    TauPoles<Statistics> u;
    MatsubaraPoles<Statistics> uhat;
    Eigen::MatrixXd fitmat;
    Eigen::BDCSVD<Eigen::MatrixXd> matrix;

    // Constructor with basis and poles
    DiscreteLehmannRepresentation(const Basis& b, const Eigen::VectorXd& poles)
        : basis(b), poles(poles),
          u(b.get_beta(), poles),
          uhat(b.get_beta(), poles)
    {
        // Fitting matrix from IR
        fitmat = -basis.s.array() * basis.v(poles).array();
        matrix.compute(fitmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }

    // Constructor with just basis
    explicit DiscreteLehmannRepresentation(const Basis& b)
        : DiscreteLehmannRepresentation(b, default_omega_sampling_points(b)) {}

    // Required virtual function implementations
    size_t size() const override { return poles.size(); }
    const Eigen::VectorXd significance() const override {
        return Eigen::VectorXd::Ones(size());
    }
    double get_accuracy() const override { return basis.get_accuracy(); }
    double get_wmax() const override { return basis.get_wmax(); }
    const Eigen::VectorXd default_tau_sampling_points() const override {
        return basis.default_tau_sampling_points();
    }

    // Convert from IR to DLR
    template <typename Derived>
    Eigen::MatrixXd from_IR(const Eigen::MatrixBase<Derived>& gl) const {
        return matrix.solve(gl);
    }

    // Convert from DLR to IR
    template <typename Derived>
    Eigen::MatrixXd to_IR(const Eigen::MatrixBase<Derived>& g_dlr) const {
        return fitmat * g_dlr;
    }

    double beta() const { return basis.beta; }
    double Lambda() const { return basis.Lambda(); }
};

} // namespace sparseir