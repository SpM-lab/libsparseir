namespace sparseir {
template <typename Statistics>
class MatsubaraPoles {
public:
    double beta;
    Eigen::VectorXd poles;

    MatsubaraPoles(double beta, const Eigen::VectorXd &poles)
        : beta(beta), poles(poles)
    {
    }

    // For Fermionic case
    template <
        typename = std::enable_if_t<std::is_same<Statistics, Fermionic>::value>>
    Eigen::VectorXd operator()(const FermionicFreq &n) const
    {
        return 1.0 / (valueim(n, beta) - poles.array()).matrix();
    }

    // For Bosonic case
    template <
        typename = std::enable_if_t<std::is_same<Statistics, Bosonic>::value>>
    Eigen::VectorXd operator()(const BosonicFreq &n) const
    {
        return (poles.array().tanh() * beta / 2.0 /
                (valueim(n, beta) - poles.array()))
            .matrix();
    }

    // For vector of frequencies
    template <typename Freq>
    Eigen::MatrixXd operator()(const std::vector<Freq> &n) const
    {
        Eigen::MatrixXd result(poles.size(), n.size());
        for (size_t i = 0; i < n.size(); ++i) {
            result.col(i) = (*this)(n[i]);
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

    TauPoles(double beta, const Eigen::VectorXd &poles)
        : beta(beta), poles(poles), omega_max(poles.array().abs().maxCoeff())
    {
    }

    Eigen::MatrixXd operator()(const Eigen::VectorXd &tau) const
    {
        // Check bounds
        for (int i = 0; i < tau.size(); ++i) {
            if (tau[i] < 0 || tau[i] > beta) {
                throw std::domain_error(
                    "τ must be in [0, β], found " + std::to_string(tau[i]) +
                    " outside of [0, " + std::to_string(beta) + "]]");
            }
        }

        // x = reshape(2τ ./ tp.β .- 1, (1, :))
        Eigen::MatrixXd x =
            (2.0 * tau.array() / beta - 1.0).matrix().transpose();

        // y = tp.poles ./ tp.ωmax
        Eigen::VectorXd y = poles.array() / omega_max;

        // Λ = tp.β * tp.ωmax
        double Lambda = beta * omega_max;

        // Create LogisticKernel
        LogisticKernel kernel(Lambda);

        // Initialize result matrix
        Eigen::MatrixXd result(poles.size(), tau.size());

        // Fill result matrix
        for (int i = 0; i < poles.size(); ++i) {
            for (int j = 0; j < tau.size(); ++j) {
                result(i, j) = -kernel(x(0, j), y(i));
            }
        }

        return result;
    }
};

template <typename Statistics, typename Basis>
class DiscreteLehmannRepresentation : public AbstractBasis<Statistics> {
public:
    Basis basis;
    Eigen::VectorXd poles;
    TauPoles<Statistics> u;
    MatsubaraPoles<Statistics> uhat;
    Eigen::MatrixXd fitmat;
    Eigen::BDCSVD<Eigen::MatrixXd> matrix;

    DiscreteLehmannRepresentation(
        const Basis &b,
        const Eigen::VectorXd &poles = default_omega_sampling_points(b))
        : basis(b), poles(poles), u(beta(b), poles), uhat(beta(b), poles)
    {
        // Fitting matrix from IR
        fitmat = -basis.s.asDiagonal() * basis.v(poles);

        // Compute SVD of fitmat
        matrix.compute(fitmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }

    size_t length() const { return poles.size(); }
    std::pair<size_t, size_t> size() const { return {poles.size(), 1}; }

    double beta() const { return basis.beta(); }
    double omega_max() const { return basis.omega_max(); }
    double Lambda() const { return basis.Lambda(); }

    const Eigen::VectorXd &sampling_points() const { return poles; }
    Eigen::VectorXd significance() const
    {
        return Eigen::VectorXd::Ones(poles.size());
    }
    double accuracy() const { return basis.accuracy(); }

    // Convert from IR to DLR
    template <typename Derived>
    Eigen::MatrixXd from_IR(const Eigen::MatrixBase<Derived> &gl,
                            int dims = 1) const
    {
        return matrix.solve(gl);
    }

    // Convert from DLR to IR
    template <typename Derived>
    Eigen::MatrixXd to_IR(const Eigen::MatrixBase<Derived> &g_dlr,
                          int dims = 1) const
    {
        return fitmat * g_dlr;
    }
};

} // namespace sparseir