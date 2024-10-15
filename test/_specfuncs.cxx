#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>
#include <iostream>
#include "sparseir/sparseir.h"
// #include "sparseir/sparseir-header-only.h"
#include <Eigen/Dense>

int main1() {
    vector<double> c = {1.0, 2.0, 3.0};
    double x = 0.5;
    double result = legval(x, c);
    std::cout << "Result of legval: " << result << std::endl;
    return 0;
}

int main2() {
    Eigen::MatrixXd c(4, 3);
    c << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12;
    int cnt = 1;
    Eigen::MatrixXd result = legder(c, cnt);
    std::cout << "Result of legder:\n" << result << std::endl;
    return 0;
}