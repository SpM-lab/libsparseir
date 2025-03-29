#include "sparseir/root.hpp"
#include "sparseir/impl/root_impl.hpp"
#include <functional>

// Forward declarations needed for specific template instantiations
namespace sparseir {
    struct Fermionic;
    struct Bosonic;
    template <typename StatType> struct MatsubaraFreq;
    template <typename StatType> class PiecewiseLegendreFT;
    class PiecewiseLegendrePoly;
    
    // Used in the implementation of sign_changes and find_extrema
    template <typename S>
    std::function<double(int)> func_for_part(const PiecewiseLegendreFT<S> &u_hat);
    
    // Forward declarations of sign_changes and find_extrema
    template <typename S>
    std::vector<MatsubaraFreq<S>> sign_changes(const PiecewiseLegendreFT<S> &u_hat, bool positive_only);
    
    template <typename S>
    std::vector<MatsubaraFreq<S>> find_extrema(const PiecewiseLegendreFT<S> &u_hat, bool positive_only);
}

// Explicit template instantiations for commonly used types
namespace sparseir {

// Explicit instantiations for midpoint
template float midpoint(float a, float b);
template double midpoint(double a, double b);
template int midpoint(int a, int b);
template double midpoint(float a, double b);
template double midpoint(double a, float b);

// Explicit instantiations for closeenough
template bool closeenough(float a, float b, float eps);
template bool closeenough(double a, double b, double eps);
template bool closeenough(int a, int b, int eps);

// Explicit instantiations for signbit
template bool signbit(float x);
template bool signbit(double x);
template bool signbit(int x);

// Explicit instantiation for refine_grid
template std::vector<float> refine_grid(const std::vector<float> &grid, int alpha);
template std::vector<double> refine_grid(const std::vector<double> &grid, int alpha);

// Explicit instantiations for find_all
template std::vector<int> find_all(std::function<double(int)>, const std::vector<int> &xgrid);
template std::vector<float> find_all(std::function<double(float)>, const std::vector<float> &xgrid);
template std::vector<double> find_all(std::function<double(double)>, const std::vector<double> &xgrid);

// Explicit instantiations for discrete_extrema
template std::vector<int> discrete_extrema(std::function<double(int)>, const std::vector<int> &xgrid);
template std::vector<float> discrete_extrema(std::function<double(float)>, const std::vector<float> &xgrid);
template std::vector<double> discrete_extrema(std::function<double(double)>, const std::vector<double> &xgrid);

// Explicit instantiations for bisect
template double bisect(const std::function<double(double)>&, double, double, double, double);


} // namespace sparseir 