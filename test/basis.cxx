#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp> // for Approx

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using Catch::Approx;

TEST_CASE("FiniteTempBasis consistency tests", "[basis]") {
    SECTION("Basic consistency") {
        double beta = 1.0;
        double omega_max = 1.0;
        double epsilon = 1e-5;
        using T = double;

        // Define the kernel
        auto kernel =sparseir::LogisticKernel(beta * omega_max);

        // Specify both template parameters: S and K
        using FermKernel = sparseir::FiniteTempBasis<sparseir::Fermionic>;
        using BosKernel = sparseir::FiniteTempBasis<sparseir::Bosonic>;

        std::pair<FermKernel, BosKernel> bases = sparseir::finite_temp_bases(beta, omega_max, epsilon);
        /*

        // Ensure FiniteTempBasisSet is properly instantiated with the kernel type
        sparseir::FiniteTempBasisSet<sparseir::LogisticKernel> bs(beta, omega_max, epsilon, kernel);

        REQUIRE(bases.first->singular_values().size() == bs.basis_f()->singular_values().size());
        REQUIRE(bases.second->singular_values().size() == bs.basis_b()->singular_values().size());

        // Clean up dynamically allocated memory if applicable
        delete bases.first;
        delete bases.second;
        */
    }

    SECTION("Sampling consistency") {
        double beta = 2.0;
        double omega_max = 5.0;
        double epsilon = 1e-5;
        using T = double;

        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        // Specify the template argument for SVEResult
        sparseir::SVEResult sve_result = sparseir::compute_sve(kernel, epsilon);

        using FermKernel = sparseir::FiniteTempBasis<sparseir::Fermionic>;
        using BosKernel = sparseir::FiniteTempBasis<sparseir::Bosonic>;

        std::pair<FermKernel, BosKernel> bases = sparseir::finite_temp_bases(beta, omega_max, epsilon, sve_result);
        /*
        sparseir::FiniteTempBasisSet<sparseir::LogisticKernel> bs(beta, omega_max, epsilon, sve_result);

        // Check sampling points consistency
        sparseir::TauSampling<sparseir::Fermionic> smpl_tau_f(bases.first);
        sparseir::TauSampling<sparseir::Bosonic> smpl_tau_b(bases.second);

        REQUIRE(smpl_tau_f.sampling_points() == smpl_tau_b.sampling_points());
        REQUIRE(bs.smpl_tau_f()->matrix() == smpl_tau_f.matrix());
        REQUIRE(bs.smpl_tau_b()->matrix() == smpl_tau_b.matrix());

        // Clean up dynamically allocated memory if applicable
        delete bases.first;
        delete bases.second;
        */
    }

    SECTION("Singular value scaling")
    {
        double beta = 1e-3;
        double omega_max = 1e-3;
        double epsilon = 1e-100;
        using T = xprec::DDouble;
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        sparseir::FiniteTempBasis<sparseir::Fermionic> basis(beta, omega_max, epsilon, kernel);
        sparseir::SVEResult sve = sparseir::compute_sve(kernel, epsilon);
        REQUIRE(sve.s.size() > 0);
        REQUIRE(basis.s.size() > 0);
        double scale = std::sqrt(beta / 2.0 * omega_max);
        // Ensure the correct function or member is used for singular values
        Eigen::VectorXd scaled_s_eigen = sve.s * scale;
        REQUIRE(basis.s.size() == sve.s.size());
        REQUIRE(basis.s.isApprox(scaled_s_eigen));
        // Access accuracy as a member variable if it's not a function
        REQUIRE(std::abs(basis.accuracy - sve.s(sve.s.size() - 1) / sve.s(0)) < 1e-10);
    }

    SECTION("Rescaling test") {
        double beta = 3.0;
        double omega_max = 4.0;
        double epsilon = 1e-6;
        using T = double;

        // Specify both template parameters
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        sparseir::FiniteTempBasis<sparseir::Fermionic> basis(beta, omega_max, epsilon, kernel);
        sparseir::FiniteTempBasis<sparseir::Fermionic> rescaled_basis = basis.rescale(2.0);
        REQUIRE(rescaled_basis.sve_result->s.size() == basis.sve_result->s.size());
        REQUIRE(rescaled_basis.get_wmax() == 6.0);
    }

    SECTION("default_sampling_points") {
        using T = double;
        auto beta = 3.0;
        auto omega_max = 4.0;
        auto epsilon = 1e-6;
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        auto basis = sparseir::FiniteTempBasis<sparseir::Fermionic>(beta, omega_max, epsilon, kernel);
        auto sve = basis.sve_result;
        auto s = sve->s;
        //REQUIRE(s.size() == 32);

        std::vector<double> s_ref = {0.5242807065966564, 0.361040299707525, 0.1600617039313169, 0.06192139783088188, 0.019641646995563183, 0.005321140031657106, 0.001245435134907047, 0.0002553808249508306, 4.6445392784931696e-5, 7.57389586542119e-6, 1.1180101601552092e-6, 1.506251988966008e-7, 1.8653991892840962e-8, 2.136773728637427e-9, 2.276179221544401e-10, 2.2655690134240947e-11, 2.115880115422964e-12, 1.861108037178489e-13, 1.5466716841180263e-14, 1.218212516630768e-15, 5.590657287253601e-16, 4.656548094642833e-16, 4.552528808131262e-16, 4.341440592462354e-16, 3.744993780121804e-16, 3.549006072192367e-16, 3.277985748785467e-16, 3.2621304578629284e-16, 3.2046691517654354e-16, 3.097729851576022e-16, 2.4973730182025644e-16, 2.476022474231314e-16};
        Eigen::VectorXd s_double = s.template cast<double>();
        REQUIRE(std::fabs(s_double[0] - s_ref[0]) < 1e-10);
        REQUIRE(std::fabs(s_double[1] - s_ref[1]) < 1e-10);
        REQUIRE(std::fabs(s_double[2] - s_ref[2]) < 1e-10);
        REQUIRE(std::fabs(s_double[3] - s_ref[3]) < 1e-10);
        REQUIRE(std::fabs(s_double[4] - s_ref[4]) < 1e-10);
        REQUIRE(std::fabs(s_double[5] - s_ref[5]) < 1e-10);
        REQUIRE(std::fabs(s_double[6] - s_ref[6]) < 1e-10);
        REQUIRE(std::fabs(s_double[7] - s_ref[7]) < 1e-10);
        REQUIRE(std::fabs(s_double[8] - s_ref[8]) < 1e-10);
        REQUIRE(std::fabs(s_double[9] - s_ref[9]) < 1e-10);
        REQUIRE(std::fabs(s_double[10] - s_ref[10]) < 1e-10);
        REQUIRE(std::fabs(s_double[11] - s_ref[11]) < 1e-10);
        REQUIRE(std::fabs(s_double[12] - s_ref[12]) < 1e-10);
        REQUIRE(std::fabs(s_double[13] - s_ref[13]) < 1e-10);
        REQUIRE(std::fabs(s_double[14] - s_ref[14]) < 1e-10);
        REQUIRE(std::fabs(s_double[15] - s_ref[15]) < 1e-10);
        REQUIRE(std::fabs(s_double[16] - s_ref[16]) < 1e-10);
        REQUIRE(std::fabs(s_double[17] - s_ref[17]) < 1e-10);
        REQUIRE(std::fabs(s_double[18] - s_ref[18]) < 1e-10);
        REQUIRE(std::fabs(s_double[19] - s_ref[19]) < 1e-10);
        //REQUIRE(std::fabs(s_double[20] - s_ref[20]) < 1e-10);
        // REQUIRE(std::fabs(s_double[21] - s_ref[21]) < 1e-10);
        // REQUIRE(std::fabs(s_double[22] - s_ref[22]) < 1e-10);

        REQUIRE(sve->u[0].data.rows() == 10);
        REQUIRE(sve->u[0].data.cols() == 32);

        std::vector<double> u_knots_ref = {
            -1.0, -0.9768276289532026, -0.9502121116288913, -0.9196860690044226,
            -0.8847415486995369, -0.8448386704449569, -0.7994218020611462, -0.7479461808659303,
            -0.6899180675604202, -0.6249508554943133, -0.552837354044473, -0.4736340017820308,
            -0.38774586460365346, -0.2959932285976203, -0.19963497739688743, -0.10032604651986517,
            0.0, 0.10032604651986517, 0.19963497739688743, 0.2959932285976203,
            0.38774586460365346, 0.4736340017820308, 0.552837354044473, 0.6249508554943133,
            0.6899180675604202, 0.7479461808659303, 0.7994218020611462, 0.8448386704449569,
            0.8847415486995369, 0.9196860690044226, 0.9502121116288913, 0.9768276289532026,
            1.0
        };
        Eigen::VectorXd u_knots_ref_eigen = Eigen::Map<Eigen::VectorXd>(u_knots_ref.data(), u_knots_ref.size());
        REQUIRE(sve->u[0].knots.isApprox(u_knots_ref_eigen));

        std::vector<double> v_knots_ref = {
            -1.0, -0.9833147686254275, -0.9470082310185116, -0.8938959515018162, -0.8283053538395936,
            -0.7548706158857138,
            -0.6778753393916265, -0.600858151891138, -0.5264593296556019, -0.45644270870032133,
            -0.39184906182935686, -0.3331494756803358, -0.2804096832724622, -0.23343248554812435,
            -0.19185635090170117, -0.15524305920516734, -0.12312152382089525,
            -0.0950206581112576, -0.070491286445028, -0.04911709970058231, -0.03050369976269751,
            -0.014178372359576086, 0.0, 0.014178372359576086, 0.03050369976269751, 0.04911709970058231,
            0.070491286445028, 0.0950206581112576, 0.12312152382089525, 0.15524305920516734, 0.19185635090170117,
            0.23343248554812435, 0.2804096832724622, 0.3331494756803358, 0.39184906182935686, 0.45644270870032133,
            0.5264593296556019, 0.600858151891138, 0.6778753393916265, 0.7548706158857138, 0.8283053538395936,
            0.8938959515018162, 0.9470082310185116, 0.9833147686254275, 1.0
        };
        Eigen::VectorXd v_knots_ref_eigen = Eigen::Map<Eigen::VectorXd>(v_knots_ref.data(), v_knots_ref.size());
        REQUIRE(sve->v[0].knots.isApprox(v_knots_ref_eigen));

        REQUIRE(sve->u[1].xmin == -1.0);
        REQUIRE(sve->u[1].xmax == 1.0);

        REQUIRE(sve->v[1].xmin == -1.0);
        REQUIRE(sve->v[1].xmax == 1.0);

        REQUIRE(sve->u[0].l == 0);
        REQUIRE(sve->u[1].l == 1);
        REQUIRE(sve->u[2].l == 2);

        REQUIRE(sve->v[0].l == 0);
        REQUIRE(sve->v[1].l == 1);
        REQUIRE(sve->v[2].l == 2);

        //REQUIRE(sve->u[1].symm == 1);
        //REQUIRE(sve->u[2].symm == -1);
        //REQUIRE(sve->u[3].symm == 1);
        //REQUIRE(sve->u[4].symm == -1);

        //REQUIRE(sve->v[1].symm == 1);
        //REQUIRE(sve->v[2].symm == -1);
        //REQUIRE(sve->v[3].symm == 1);
        //REQUIRE(sve->v[4].symm == -1);

        //std::cout << "Singular values: " << s.transpose() << std::endl;
        /*
        int L = 10;
        Eigen::VectorXd pts_L = default_sampling_points(sve->u, L);
        REQUIRE(pts_L.size() == L);
        Eigen::VectorXd pts_100 = default_sampling_points(sve->u, 100);
        REQUIRE(pts_100.size() == 24);
        */
    }
}
