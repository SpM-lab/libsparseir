#include <iostream>
#include <Eigen/Core>
int main( ) {
    Eigen::Vector3d v( 0.0, 0.0, 1.0 );
    Eigen::Matrix3d m;
    m << 0.0, -1.0, -1.0, 1.0, 0.0, -2.0, 0.0, 0.0, 1.0;
    std::cout << m * v << std::endl;
    return ( 0 );
}

