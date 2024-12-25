#pragma once

#include <memory>
#include <Eigen/Dense>

namespace sparseir {

template <typename T>
class FiniteTempBasisSet {
public:
    using Scalar = T;
    using BasisPtr = std::shared_ptr<AbstractBasis<Scalar>>;
    using TauSamplingPtr = std::shared_ptr<TauSampling<Scalar>>;
    using MatsubaraSamplingPtr = std::shared_ptr<MatsubaraSampling<std::complex<Scalar>>>;

    // Constructors
    FiniteTempBasisSet(Scalar beta, Scalar omega_max, Scalar epsilon = std::numeric_limits<Scalar>::quiet_NaN(),
                       const SVEResult& sve_result = SVEResult())
        : beta_(beta), omega_max_(omega_max), epsilon_(epsilon) {
        initialize(sve_result);
    }

    // Accessors
    Scalar beta() const { return beta_; }
    Scalar omega_max() const { return omega_max_; }
    Scalar accuracy() const { return epsilon_; }

    const BasisPtr& basis_f() const { return basis_f_; }
    const BasisPtr& basis_b() const { return basis_b_; }

    const TauSamplingPtr& smpl_tau_f() const { return smpl_tau_f_; }
    const TauSamplingPtr& smpl_tau_b() const { return smpl_tau_b_; }

    const MatsubaraSamplingPtr& smpl_wn_f() const { return smpl_wn_f_; }
    const MatsubaraSamplingPtr& smpl_wn_b() const { return smpl_wn_b_; }

    const Eigen::VectorXd& tau() const { return smpl_tau_f_->sampling_points(); }
    const std::vector<int>& wn_f() const { return smpl_wn_f_->sampling_frequencies(); }
    const std::vector<int>& wn_b() const { return smpl_wn_b_->sampling_frequencies(); }

    const SVEResult& sve_result() const { return sve_result_; }

private:
     void initialize(const SVEResult& sve_result_input) {
        if (std::isnan(epsilon_)) {
            epsilon_ = std::numeric_limits<Scalar>::epsilon();
        }

        LogisticKernel kernel(beta_ * omega_max_);
        sve_result_ = sve_result_input.is_valid() ? sve_result_input : compute_sve(kernel);

        basis_f_ = std::make_shared<FiniteTempBasis<Fermionic, LogisticKernel>>(
            beta_, omega_max_, epsilon_, kernel, sve_result_);
        basis_b_ = std::make_shared<FiniteTempBasis<Bosonic, LogisticKernel>>(
            beta_, omega_max_, epsilon_, kernel, sve_result_);

        // Initialize sampling objects
        smpl_tau_f_ = std::make_shared<TauSampling<Scalar>>(basis_f_);
        smpl_tau_b_ = std::make_shared<TauSampling<Scalar>>(basis_b_);

        smpl_wn_f_ = std::make_shared<MatsubaraSampling<std::complex<Scalar>>>(basis_f_);
        smpl_wn_b_ = std::make_shared<MatsubaraSampling<std::complex<Scalar>>>(basis_b_);
    }

    Scalar beta_;
    Scalar omega_max_;
    Scalar epsilon_;

    BasisPtr basis_f_;
    BasisPtr basis_b_;

    TauSamplingPtr smpl_tau_f_;
    TauSamplingPtr smpl_tau_b_;

    MatsubaraSamplingPtr smpl_wn_f_;
    MatsubaraSamplingPtr smpl_wn_b_;

    SVEResult sve_result_;
};

} // namespace sparseir