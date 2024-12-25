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

template<typename T>
void
_test_sve() {
    auto lk = sparseir::LogisticKernel(10.0);
    REQUIRE(lk.lambda_ == 10.0);
    auto hints = sparseir::sve_hints<T>(lk, 1e-6);
    int nsvals_hint = hints.nsvals();
    int n_gauss = hints.ngauss();
    std::vector<T> segs_x = hints.segments_x();
    std::vector<T> segs_y = hints.segments_y();

    // Ensure `convert` is declared before this line
    sparseir::Rule<T> rule = sparseir::legendre<T>(n_gauss);

    sparseir::Rule<T> gauss_x = rule.piecewise(segs_x);
    sparseir::Rule<T> gauss_y = rule.piecewise(segs_y);
    auto ssve1 = sparseir::SamplingSVE<sparseir::LogisticKernel,T>(lk, 1e-6);
    REQUIRE(ssve1.n_gauss == n_gauss);
    auto ssve2 = sparseir::SamplingSVE<sparseir::LogisticKernel,T>(lk, 1e-6, 12);
    REQUIRE(ssve2.n_gauss == 12);

    auto ssve1_double =
        sparseir::SamplingSVE<sparseir::LogisticKernel, T>(lk, 1e-6);
    REQUIRE(ssve1_double.n_gauss == n_gauss);
    auto ssve2_double =
        sparseir::SamplingSVE<sparseir::LogisticKernel, T>(lk, 1e-6, 12);
    REQUIRE(ssve2_double.n_gauss == 12);
}

TEST_CASE("sve.cpp", "[SamplingSVE]")
{
    _test_sve<double>();
    _test_sve<xprec::DDouble>();
}


template<typename T>
void
_test_centrosymmsve() {
    auto lk = sparseir::LogisticKernel(10.0);
    auto hints = sparseir::sve_hints<T>(lk, 1e-6);
    int nsvals_hint = hints.nsvals();
    int n_gauss = hints.ngauss();
    std::vector<T> segs_x = hints.segments_x();
    std::vector<T> segs_y = hints.segments_y();

    sparseir::Rule<T> rule = sparseir::legendre<T>(n_gauss);

    auto sve = sparseir::CentrosymmSVE<sparseir::LogisticKernel,T>(lk, 1e-6);
    // any check?
}

TEST_CASE("CentrosymmSVE", "[CentrosymmSVE]")
{
    _test_centrosymmsve<double>();
    _test_centrosymmsve<xprec::DDouble>();
}

TEST_CASE("compute_sve", "[compute_sve]"){
    auto epsilon = std::numeric_limits<double>::quiet_NaN();

    double safe_epsilon;
    std::string Twork_actual;
    std::string svd_strategy_actual;
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) = sparseir::auto_choose_accuracy(epsilon, "Float64x2");

    REQUIRE(Twork_actual == "Float64x2");

    using T = xprec::DDouble;
    auto lk = sparseir::LogisticKernel(12.0);

    //auto sve = sparseir::compute_sve<sparseir::LogisticKernel, T>(lk, safe_epsilon);
    auto sve = sparseir::compute_sve<sparseir::LogisticKernel>(lk, safe_epsilon);
    auto s = sve.s;
    //std::cout << "S values: \n" << s << std::endl;
    //std::cout << "diff " << s[0] - 0.52428 << std::endl;
    auto diff = s[0] - 0.5242807065966566;
    REQUIRE(std::abs(diff) < 1e-20);
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
    REQUIRE(sparseir::auto_choose_accuracy(1e-6, "Float64", "auto") ==
            std::make_tuple(1.0e-6, "Float64", "default"));
    REQUIRE(sparseir::auto_choose_accuracy(1e-6, "Float64", "accurate") ==
            std::make_tuple(1.0e-6, "Float64", "accurate"));

}

TEST_CASE("sve.cpp", "[truncate]")
{
    using T = double;
    sparseir::CentrosymmSVE<sparseir::LogisticKernel, T> sve(sparseir::LogisticKernel(5.), 1e-6);
    std::vector<Eigen::MatrixX<T>> matrices = sve.matrices();
    REQUIRE(matrices.size() == 2);
    std::vector<std::tuple<Eigen::MatrixX<T>, Eigen::MatrixX<T>,
                            Eigen::MatrixX<T>>>
        svds;
    for (const auto &mat : matrices) {
        auto svd = sparseir::compute_svd(mat);
        svds.push_back(svd);
    }


    // Extract singular values and vectors
    std::vector<Eigen::MatrixX<T>> u_list, v_list;
    std::vector<Eigen::VectorX<T>> s_list;
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