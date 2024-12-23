// sparseir/tests/sve.cpp

#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using std::invalid_argument;

// Function to check smoothness
void check_smooth(const std::function<double(double)> &u,
                  const std::vector<double> &s, double uscale,
                  double fudge_factor)
{
    /*
    double epsilon = std::numeric_limits<double>::epsilon();
    std::vector<double> knots = sparseir::knots(u);

    // Remove the first and last knots
    if (knots.size() <= 2) {
        REQUIRE(false); // Not enough knots to perform the test
    }
    knots.erase(knots.begin());
    knots.pop_back();

    std::vector<double> jump(knots.size());
    std::vector<double> compare_below(knots.size());
    std::vector<double> compare_above(knots.size());
    std::vector<double> compare(knots.size());

    for (size_t i = 0; i < knots.size(); ++i) {
        double x = knots[i];
        jump[i] = std::abs(u(x + epsilon) - u(x - epsilon));
        compare_below[i] = std::abs(u(x - epsilon) - u(x - 3 * epsilon));
        compare_above[i] = std::abs(u(x + 3 * epsilon) - u(x + epsilon));
        compare[i] = std::min(compare_below[i], compare_above[i]);
        compare[i] = std::max(compare[i], uscale * epsilon);
        // Loss of precision
        compare[i] *= fudge_factor * (s[0] / s[i]);
    }

    bool all_less = true;
    for (size_t i = 0; i < jump.size(); ++i) {
        if (!(jump[i] < compare[i])) {
            all_less = false;
            break;
        }
    }

    REQUIRE(all_less);
    */
}

TEST_CASE("sve.cpp", "[SamplingSVE]")
{
    sparseir::LogisticKernel lk(10.0);
    REQUIRE(lk.lambda_ == 10.0);
    auto hints = sparseir::sve_hints(lk, 1e-6);
    int nsvals_hint = hints.nsvals();
    int n_gauss = hints.ngauss();
    std::vector<double> segs_x = hints.template segments_x<double>();
    std::vector<double> segs_y = hints.template segments_y<double>();

    // Ensure `convert` is declared before this line
    sparseir::Rule<double> rule = sparseir::legendre<double>(n_gauss);

    sparseir::Rule<double> gauss_x = rule.piecewise(segs_x);
    sparseir::Rule<double> gauss_y = rule.piecewise(segs_y);
    auto ssve1 = sparseir::SamplingSVE<sparseir::LogisticKernel>(lk, 1e-6);
    REQUIRE(ssve1.n_gauss == n_gauss);
    auto ssve2 = sparseir::SamplingSVE<sparseir::LogisticKernel>(lk, 1e-6, 12);
    REQUIRE(ssve2.n_gauss == 12);

    auto ssve1_double =
        sparseir::SamplingSVE<sparseir::LogisticKernel, double>(lk, 1e-6);
    REQUIRE(ssve1_double.n_gauss == n_gauss);
    auto ssve2_double =
        sparseir::SamplingSVE<sparseir::LogisticKernel, double>(lk, 1e-6, 12);
    REQUIRE(ssve2_double.n_gauss == 12);

    auto ssve1_ddouble =
        sparseir::SamplingSVE<sparseir::LogisticKernel, xprec::DDouble>(lk,
                                                                        1e-6);
    REQUIRE(ssve1_ddouble.n_gauss == n_gauss);
    auto ssve2_ddouble =
        sparseir::SamplingSVE<sparseir::LogisticKernel, xprec::DDouble>(
            lk, 1e-6, 12);
    REQUIRE(ssve2_ddouble.n_gauss == 12);
}

TEST_CASE("CentrosymmSVE", "[CentrosymmSVE]")
{
    sparseir::LogisticKernel lk(10.0);
    auto hints = sparseir::sve_hints(lk, 1e-6);
    int nsvals_hint = hints.nsvals();
    int n_gauss = hints.ngauss();
    std::vector<double> segs_x = hints.template segments_x<double>();
    std::vector<double> segs_y = hints.template segments_y<double>();

    sparseir::Rule<double> rule = sparseir::legendre<double>(n_gauss);

    auto sve = sparseir::CentrosymmSVE<sparseir::LogisticKernel>(lk, 1e-6);
    auto sve_double =
        sparseir::CentrosymmSVE<sparseir::LogisticKernel, double>(lk, 1e-6);
    auto sve_ddouble =
        sparseir::CentrosymmSVE<sparseir::LogisticKernel, xprec::DDouble>(lk,
                                                                          1e-6);
}

TEST_CASE("compute_sve", "[compute_sve]"){
    sparseir::LogisticKernel lk(12.0);
    auto sve = sparseir::compute_sve<sparseir::LogisticKernel>(lk);
    auto s = sve.s;
    //std::cout << "S values: \n" << s << std::endl;
}

TEST_CASE("sve.cpp", "[compute_sve]")
{

    // Define a map to store SVEResult objects
    //auto sve_logistic = std::map < int,
    //sparseir::SVEResult<sparseir::LogisticKernel>>{
      //                                          {10,
      //                                          sparseir::compute_sve<sparseir::LogisticKernel>(sparseir::LogisticKernel(10.0))},
      //                                          {42,
      //                                          sparseir::compute_sve<sparseir::LogisticKernel>(sparseir::LogisticKernel(42.0))},
      //                                          {10000,
      //                                          sparseir::compute_sve<sparseir::LogisticKernel>(sparseir::LogisticKernel(10000.0))},
      //                                          {100000000,
      //                                          sparseir::compute_sve<sparseir::LogisticKernel>(sparseir::LogisticKernel(10000.0),
      //                                          1e-12)},
      //                                          };

    SECTION("smooth with Λ =")
    {
        for (int Lambda : {10, 42, 10000}) {
            REQUIRE(true);
            // sparseir::FiniteTempBasis<sparseir::Fermionic,
            // sparseir::LogisticKernel> basis(1, Lambda, sve_logistic[Lambda]);
            // TODO: Check that the maximum implementation  is defined
            // check_smooth(basis.u, basis.s, 2 * sparseir::maximum(basis.u(1)),
            // 24);
            // check_smooth(basis.v, basis.s, 50, 200);
        }
    }

    /*
    SECTION("num roots u with Λ =") {
        for (int Lambda : {10, 42, 10000}) {
            FiniteTempBasis<Fermionic> basis(1, Lambda,
    sparseir::sve_logistic[Lambda]); for (const auto& ui : basis.u) {
                std::vector<double> ui_roots = sparseir::roots(ui);
                REQUIRE(ui_roots.size() == static_cast<size_t>(ui.l));
            }
        }
    }
    */

    /*
    SECTION("num roots û with stat =, Λ =") {
        for (const auto& stat : {Fermionic(), Bosonic()}) {
            for (int Lambda : {10, 42, 10000}) {
                FiniteTempBasis basis(stat, 1, Lambda,
    sparseir::sve_logistic[Lambda]); for (int i : {1, 2, 8, 11}) {
                    std::vector<double> x0 =
    sparseir::find_extrema(basis.uhat[i]); REQUIRE(i <= x0.size() && x0.size()
    <= static_cast<size_t>(i + 1));
                }
            }
        }
    }
    */

    /*
    SECTION("accuracy with stat =, Λ =") {
        for (const auto& stat : {Fermionic(), Bosonic()}) {
            for (int Lambda : {10, 42, 10000}) {
                FiniteTempBasis basis(stat, 4, Lambda,
    sparseir::sve_logistic[Lambda]); REQUIRE(sparseir::accuracy(basis) <=
    sparseir::significance(basis).back());
                REQUIRE(sparseir::significance(basis).front() == 1.0);
                REQUIRE(sparseir::accuracy(basis) <= (basis.s.back() /
    basis.s.front()));
            }
        }
    }
    */
}

TEST_CASE("sve.cpp", "[choose_accuracy]")
{
    REQUIRE(sparseir::choose_accuracy(nullptr, nullptr) ==
            std::make_tuple(2.2204460492503131e-16, "Float64x2", "default"));
    REQUIRE(sparseir::choose_accuracy(nullptr, "Float64") ==
            std::make_tuple(1.4901161193847656e-8, "Float64", "default"));
    REQUIRE(sparseir::choose_accuracy(nullptr, "Float64x2") ==
            std::make_tuple(2.2204460492503131e-16, "Float64x2", "default"));

    REQUIRE(sparseir::choose_accuracy(1e-6, nullptr) ==
            std::make_tuple(1.0e-6, "Float64", "default"));
    // Note: Catch2 doesn't have a built-in way to capture logs.
    // You might need to implement a custom logger or use a library that
    // supports log capturing. Add debug output to see the actual return value
    REQUIRE(sparseir::choose_accuracy(1e-8, nullptr) ==
            std::make_tuple(1.0e-8, "Float64x2", "default"));
    REQUIRE(sparseir::choose_accuracy(1e-20, nullptr) ==
            std::make_tuple(1.0e-20, "Float64x2", "default"));

    REQUIRE(sparseir::choose_accuracy(1e-10, "Float64") ==
            std::make_tuple(1.0e-10, "Float64", "accurate"));

    REQUIRE(sparseir::choose_accuracy(1e-6, "Float64") ==
            std::make_tuple(1.0e-6, "Float64", "default"));
    REQUIRE(sparseir::choose_accuracy(1e-6, "Float64", "auto") ==
            std::make_tuple(1.0e-6, "Float64", "default"));
    REQUIRE(sparseir::choose_accuracy(1e-6, "Float64", "accurate") ==
            std::make_tuple(1.0e-6, "Float64", "accurate"));

}

TEST_CASE("sve.cpp", "[truncate]")
{
    sparseir::CentrosymmSVE<sparseir::LogisticKernel, double> sve(
    sparseir::LogisticKernel(5), 1e-6);
    std::vector<Eigen::MatrixX<double>> matrices = sve.matrices();
    REQUIRE(matrices.size() == 2);
    std::vector<std::tuple<Eigen::MatrixX<double>, Eigen::MatrixX<double>,
                            Eigen::MatrixX<double>>>
        svds;
    for (const auto &mat : matrices) {
        auto svd = sparseir::compute_svd(mat);
        svds.push_back(svd);
    }


    // Extract singular values and vectors
    std::vector<Eigen::MatrixX<double>> u_list, v_list;
    std::vector<Eigen::VectorX<double>> s_list;
    for (const auto &svd : svds) {
        auto u = std::get<0>(svd);
        auto s = std::get<1>(svd);
        auto v = std::get<2>(svd);
        u_list.push_back(u);
        s_list.push_back(s);
        v_list.push_back(v);
    }
    for (int lmax = 3; lmax <= 20; ++lmax) {
        auto truncated =
            sparseir::truncate(u_list, s_list, v_list, 1e-8, lmax);
        auto u = std::get<0>(truncated);
        auto s = std::get<1>(truncated);
        auto v = std::get<2>(truncated);

        auto sveresult = sve.postprocess(u, s, v);
        REQUIRE(sveresult.u.size() == sveresult.s.size());
        REQUIRE(sveresult.s.size() == sveresult.v.size());
        REQUIRE(sveresult.s.size() <= static_cast<size_t>(lmax - 1));
    }
}
