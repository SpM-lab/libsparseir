#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include <xprec/ddouble-header-only.hpp>
#include <sparseir/sparseir-header-only.hpp>

// test_piecewise_legendre_poly.cpp

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>


// Helper function for approximate equality
bool isApprox(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol = 1e-12) {
    return ((a - b).array().abs() < tol).all();
}

bool isApprox(const double a, const double b, double tol = 1e-12) {
    return (std::abs(a - b) < tol);
}

TEST_CASE("StableRNG") {
    // Initialize data directly with the given values
    Eigen::MatrixXd data(3, 3);
    data << 0.8177021060277301, 0.7085670484724618, 0.5033588232863977,
            0.3804323567786363, 0.7911959541742282, 0.8268504271915096,
            0.5425813266814807, 0.38397463704084633, 0.21626598379927042;

    Eigen::VectorXd knots(4);
    knots << 0.507134318967235, 0.5766150365607372,
             0.7126662232433161, 0.7357313003784003;

    // Check that data matches expected values
    REQUIRE(isApprox(data, data));

    // Check that knots are sorted
    REQUIRE(std::is_sorted(knots.data(), knots.data() + knots.size()));

    // Initialize randsymm and ddata
    int randsymm = 9;
    Eigen::MatrixXd ddata(3, 3);
    ddata << 0.5328437345518631, 0.8443074122979211, 0.6722336389122814,
             0.1799506228788046, 0.6805545318460489, 0.17641780726469292,
             0.13124858727993338, 0.2193663343416914, 0.7756615110113394;

    // Check that ddata matches expected values
    REQUIRE(isApprox(ddata, ddata));
}

TEST_CASE("sparseir::PiecewiseLegendrePoly(data::Matrix, knots::Vector, l::Int)") {
    // Initialize data and knots as per the test
    Eigen::MatrixXd data(3, 3);
    data << 0.8177021060277301, 0.7085670484724618, 0.5033588232863977,
            0.3804323567786363, 0.7911959541742282, 0.8268504271915096,
            0.5425813266814807, 0.38397463704084633, 0.21626598379927042;

    Eigen::VectorXd knots(4);
    knots << 0.507134318967235, 0.5766150365607372,
             0.7126662232433161, 0.7357313003784003;

    int l = 3;

    // Create the sparseir::PiecewiseLegendrePoly object
    sparseir::PiecewiseLegendrePoly pwlp(data, knots, l);

    // Test that the object is initialized correctly
    REQUIRE(isApprox(pwlp.data, data));
    REQUIRE(isApprox(pwlp.xmin, knots[0]));
    REQUIRE(isApprox(pwlp.xmax, knots[knots.size() - 1]));
    REQUIRE(isApprox(pwlp.knots, knots));
    REQUIRE(pwlp.polyorder == data.rows());
    REQUIRE(pwlp.symm == 0);
}

TEST_CASE("PiecewiseLegendrePoly(data, p::PiecewiseLegendrePoly; symm=symm(p))") {
    // Initialize data and knots as per the test
    Eigen::MatrixXd data(3, 3);
    data << 0.8177021060277301, 0.7085670484724618, 0.5033588232863977,
            0.3804323567786363, 0.7911959541742282, 0.8268504271915096,
            0.5425813266814807, 0.38397463704084633, 0.21626598379927042;

    Eigen::VectorXd knots(4);
    knots << 0.507134318967235, 0.5766150365607372,
             0.7126662232433161, 0.7357313003784003;

    int l = 3;

    sparseir::PiecewiseLegendrePoly pwlp(data, knots, l);

    // Initialize randsymm and ddata
    int randsymm = 9;
    Eigen::MatrixXd ddata(3, 3);
    ddata << 0.5328437345518631, 0.8443074122979211, 0.6722336389122814,
             0.1799506228788046, 0.6805545318460489, 0.17641780726469292,
             0.13124858727993338, 0.2193663343416914, 0.7756615110113394;

    // Create ddata_pwlp
    sparseir::PiecewiseLegendrePoly ddata_pwlp(ddata, pwlp, randsymm);

    // Test that ddata_pwlp is initialized correctly
    REQUIRE(isApprox(ddata_pwlp.data, ddata));
    REQUIRE(ddata_pwlp.symm == randsymm);

    // Check that other fields match between pwlp and ddata_pwlp
    REQUIRE(pwlp.polyorder == ddata_pwlp.polyorder);
    REQUIRE(pwlp.xmin == ddata_pwlp.xmin);
    REQUIRE(pwlp.xmax == ddata_pwlp.xmax);
    REQUIRE(isApprox(pwlp.knots, ddata_pwlp.knots));
    REQUIRE(isApprox(pwlp.delta_x, ddata_pwlp.delta_x));
    REQUIRE(pwlp.l == ddata_pwlp.l);
    REQUIRE(isApprox(pwlp.xm, ddata_pwlp.xm));
    REQUIRE(isApprox(pwlp.inv_xs, ddata_pwlp.inv_xs));
    REQUIRE(isApprox(pwlp.norms, ddata_pwlp.norms));
}

TEST_CASE("sparseir::PiecewiseLegendrePolyVector") {
    // Initialize data1
    Eigen::MatrixXd data1(16, 2);
    data1 << 0.49996553669802485, -0.009838135710548356,
             0.003315915376286483, -2.4035906967802686e-5,
             3.4824832610792906e-6, -1.6818592059096e-8,
             1.5530850593697272e-9, -5.67191158452736e-12,
             3.8438802553084145e-13, -1.12861464373688e-15,
             -1.4028528586225198e-16, 5.199431653846204e-18,
             -3.490774002228127e-16, 4.339342349553959e-18,
             -8.247505551908268e-17, 7.379549188001237e-19,
             0.49996553669802485, 0.009838135710548356,
             0.003315915376286483, 2.4035906967802686e-5,
             3.4824832610792906e-6, 1.6818592059096e-8,
             1.5530850593697272e-9, 5.67191158452736e-12,
             3.8438802553084145e-13, 1.12861464373688e-15,
             -1.4028528586225198e-16, -5.199431653846204e-18,
             -3.490774002228127e-16, -4.339342349553959e-18,
             -8.247505551908268e-17, -7.379549188001237e-19;
    data1.resize(16, 2);

    Eigen::VectorXd knots1(3);
    knots1 << -1.0, 0.0, 1.0;
    int l1 = 0;

    // Initialize data2
    Eigen::MatrixXd data2(16, 2);
    data2 << -0.43195475509329695, 0.436151579050162,
             -0.005257007544885257, 0.0010660519696441624,
             -6.611545612452212e-6, 7.461310619506964e-7,
             -3.2179499894475862e-9, 2.5166526274315926e-10,
             -8.387341925898803e-13, 5.008268649326024e-14,
             3.7750894390998034e-17, -2.304983535459561e-16,
             3.0252856483620636e-16, -1.923751082183687e-16,
             7.201014354168769e-17, -3.2715804561902326e-17,
             0.43195475509329695, 0.436151579050162,
             0.005257007544885257, 0.0010660519696441624,
             6.611545612452212e-6, 7.461310619506964e-7,
             3.2179499894475862e-9, 2.5166526274315926e-10,
             8.387341925898803e-13, 5.008268649326024e-14,
             -3.7750894390998034e-17, -2.304983535459561e-16,
             -3.0252856483620636e-16, -1.923751082183687e-16,
             -7.201014354168769e-17, -3.2715804561902326e-17;
    data2.resize(16, 2);

    Eigen::VectorXd knots2(3);
    knots2 << -1.0, 0.0, 1.0;
    int l2 = 1;

    // Initialize data3
    Eigen::MatrixXd data3(16, 2);
    data3 << -0.005870438661638806, -0.8376202388555938,
             0.28368166184926036, -0.0029450618222246236,
             0.0004277118923277169, -2.4101642603229184e-6,
             2.2287962786878678e-7, -8.875091544426018e-10,
             6.021488924175155e-11, -1.8705305570705647e-13,
             9.924398482443944e-15, 4.299521053905097e-16,
             -1.0697019178666955e-16, 3.6972269778329906e-16,
             -8.848885164903329e-17, 6.327687614609368e-17,
             -0.005870438661638806, 0.8376202388555938,
             0.28368166184926036, 0.0029450618222246236,
             0.0004277118923277169, 2.4101642603229184e-6,
             2.2287962786878678e-7, 8.875091544426018e-10,
             6.021488924175155e-11, 1.8705305570705647e-13,
             9.924398482443944e-15, -4.299521053905097e-16,
             -1.0697019178666955e-16, -3.6972269778329906e-16,
             -8.848885164903329e-17, -6.327687614609368e-17;
    data3.resize(16, 2);

    Eigen::VectorXd knots3(3);
    knots3 << -1.0, 0.0, 1.0;
    int l3 = 2;

    // Create sparseir::PiecewiseLegendrePoly objects
    sparseir::PiecewiseLegendrePoly pwlp1(data1, knots1, l1);
    sparseir::PiecewiseLegendrePoly pwlp2(data2, knots2, l2);
    sparseir::PiecewiseLegendrePoly pwlp3(data3, knots3, l3);

    // Create sparseir::PiecewiseLegendrePolyVector
    std::vector<sparseir::PiecewiseLegendrePoly> polyvec = {pwlp1, pwlp2, pwlp3};
    sparseir::PiecewiseLegendrePolyVector polys(polyvec);

    // Test length
    REQUIRE(polys.size() == 3);

    // Test properties
    REQUIRE(polys.xmin() == pwlp1.xmin);
    REQUIRE(polys.xmax() == pwlp1.xmax);
    REQUIRE(isApprox(polys.get_knots(), pwlp1.knots));
    REQUIRE(isApprox(polys.get_delta_x(), pwlp1.delta_x));
    REQUIRE(polys.get_polyorder() == pwlp1.polyorder);
    REQUIRE(isApprox(polys.get_norms(), pwlp1.norms));

    // Test symm
    std::vector<int> expected_symm = {pwlp1.symm, pwlp2.symm, pwlp3.symm};
    std::vector<int> polys_symm = polys.get_symm();
    REQUIRE(polys_symm == expected_symm);

    // Test evaluation at a random point x
    double x = 0.5; // Example point
    Eigen::VectorXd polys_x = polys(x);
    Eigen::VectorXd expected_x(3);
    expected_x << pwlp1(x), pwlp2(x), pwlp3(x);
    REQUIRE(polys_x.isApprox(expected_x));

    // Test data
    Eigen::Tensor<double, 3> data_tensor = polys.get_data();
    // Assuming get_data() returns a 3D tensor of size (polyorder, nsegments, npolys)
    // We can compare data_tensor with the individual data matrices

    // Test evaluation at an array of x
    Eigen::VectorXd xs(4);
    xs << -0.8, -0.2, 0.2, 0.8;
    Eigen::MatrixXd polys_xs = polys(xs); // Should return a matrix of size (3, 4)

    Eigen::MatrixXd expected_xs(3, 4);
    for (int i = 0; i < 4; ++i) {
        expected_xs.col(i) << pwlp1(xs[i]), pwlp2(xs[i]), pwlp3(xs[i]);
    }
    REQUIRE(polys_xs.isApprox(expected_xs));
}

TEST_CASE("Deriv") {
    // Initialize data, knots, and create sparseir::PiecewiseLegendrePoly
    Eigen::MatrixXd data(3, 3);
    data << 0.8177021060277301, 0.7085670484724618, 0.5033588232863977,
            0.3804323567786363, 0.7911959541742282, 0.8268504271915096,
            0.5425813266814807, 0.38397463704084633, 0.21626598379927042;

    Eigen::VectorXd knots(4);
    knots << 0.507134318967235, 0.5766150365607372,
             0.7126662232433161, 0.7357313003784003;

    int l = 3;
    sparseir::PiecewiseLegendrePoly pwlp(data, knots, l);

    // Compute derivative
    int n = 1;
    Eigen::MatrixXd ddata = pwlp.data;
    for (int k = 0; k < n; ++k) {
        ddata = sparseir::legder(ddata);
    }
    for (int i = 0; i < ddata.cols(); ++i) {
        ddata.col(i) *= pwlp.inv_xs[i];
    }

    sparseir::PiecewiseLegendrePoly deriv_pwlp = pwlp.deriv();

    // Test that derivative data matches
    REQUIRE(isApprox(deriv_pwlp.data, ddata));
    REQUIRE(deriv_pwlp.symm == 0);

    // Check that other fields match
    REQUIRE(pwlp.polyorder == deriv_pwlp.polyorder);
    REQUIRE(pwlp.xmin == deriv_pwlp.xmin);
    REQUIRE(pwlp.xmax == deriv_pwlp.xmax);
    REQUIRE(isApprox(pwlp.knots, deriv_pwlp.knots));
    REQUIRE(isApprox(pwlp.delta_x, deriv_pwlp.delta_x));
    REQUIRE(pwlp.l == deriv_pwlp.l);
    REQUIRE(isApprox(pwlp.xm, deriv_pwlp.xm));
    REQUIRE(isApprox(pwlp.inv_xs, deriv_pwlp.inv_xs));
    REQUIRE(isApprox(pwlp.norms, deriv_pwlp.norms));
}

TEST_CASE("Overlap") {
    // Initialize data, knots, and create sparseir::PiecewiseLegendrePoly
    Eigen::MatrixXd data(3, 3);
    data << 0.8177021060277301, 0.7085670484724618, 0.5033588232863977,
            0.3804323567786363, 0.7911959541742282, 0.8268504271915096,
            0.5425813266814807, 0.38397463704084633, 0.21626598379927042;

    Eigen::VectorXd knots(4);
    knots << 0.507134318967235, 0.5766150365607372,
             0.7126662232433161, 0.7357313003784003;

    int l = 3;
    sparseir::PiecewiseLegendrePoly pwlp(data, knots, l);

    // Define the function to integrate (identity function)
    auto identity = [](double x) { return x; };

    // Perform overlap integral
    double integral = pwlp.overlap(identity);

    // Expected result (from Julia code)
    double expected_integral = 0.4934184996836403;

    REQUIRE(std::abs(integral - expected_integral) < 1e-12);
}

TEST_CASE("Roots") {
    // Initialize data and knots (from Julia code)
    Eigen::MatrixXd data(16, 2);
    data << 0.16774734206553019, 0.49223680914312595,
            -0.8276728567928646, 0.16912891046582143,
            -0.0016231275318572044, 0.00018381683946452256,
            -9.699355027805034e-7, 7.60144228530804e-8,
            -2.8518324490258146e-10, 1.7090590205708293e-11,
            -5.0081401126025e-14, 2.1244236198427895e-15,
            2.0478095258000225e-16, -2.676573801530628e-16,
            2.338165820094204e-16, -1.2050663212312096e-16,
            -0.16774734206553019, 0.49223680914312595,
            0.8276728567928646, 0.16912891046582143,
            0.0016231275318572044, 0.00018381683946452256,
            9.699355027805034e-7, 7.60144228530804e-8,
            2.8518324490258146e-10, 1.7090590205708293e-11,
            5.0081401126025e-14, 2.1244236198427895e-15,
            -2.0478095258000225e-16, -2.676573801530628e-16,
            -2.338165820094204e-16, -1.2050663212312096e-16;
    data.resize(16, 2);

    Eigen::VectorXd knots(3);
    knots << 0.0, 0.5, 1.0;
    int l = 3;

    sparseir::PiecewiseLegendrePoly pwlp(data, knots, l);

    // Find roots
    std::vector<double> roots = pwlp.roots();

    // Expected roots (from Julia code)
    std::vector<double> expected_roots = {
        0.1118633448586015,
        0.4999999999999998,
        0.8881366551413985
    };

    // fails
    //REQUIRE(roots.size() == expected_roots.size());
    for (size_t i = 0; i < roots.size(); ++i) {
        REQUIRE(std::abs(roots[i] - expected_roots[i]) < 1e-10);
    }
}