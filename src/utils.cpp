#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <numeric>

namespace sparseir {

// julia> sort = sortperm(s; rev=true)
// Implement sortperm in C++
std::vector<size_t> sortperm_rev(const Eigen::VectorXd &vec)
{
    std::vector<size_t> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; });
    return indices;
}

} // namespace sparseir