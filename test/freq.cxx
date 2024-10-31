#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <xprec/ddouble-header-only.hpp>
#include <sparseir/sparseir-header-only.hpp>

using std::invalid_argument;

using xprec::DDouble;
using Eigen::Matrix;

TEST_CASE("gauss", "[Rule]")
{
    // Initialize x, w, v, x_forward, x_backward with DDouble values
    std::vector<DDouble> x, w;
    Rule<DDouble> r = Rule<DDouble>(x, w);
    REQUIRE(1==1);
}


template <typename T>
void gaussValidate(const Rule<T>& rule) {
    if (!(rule.a <= rule.b)) {
        throw invalid_argument("a,b must be a valid interval");
    }
    if (!all_of(rule.x.begin(), rule.x.end(), [rule](T xi) { return xi <= rule.b; })) {
        throw invalid_argument("x must be smaller than b");
    }
    if (!all_of(rule.x.begin(), rule.x.end(), [rule](T xi) { return xi >= rule.a; })) {
        throw invalid_argument("x must be larger than a");
    }
    if (!is_sorted(rule.x.begin(), rule.x.end())) {
        throw invalid_argument("x must be well-ordered");
    }
    if (rule.x.size() != rule.w.size()) {
        throw invalid_argument("shapes are inconsistent");
    }

    // TODO: Fix me
    //REQUIRE(equal(rule.x_forward.begin(), rule.x_forward.end(), rule.x.begin(), [rule](T xi, T x_forward) { return abs(x_forward - (xi - rule.a)) < 1e-9; }));
    //REQUIRE(equal(rule.x_backward.begin(), rule.x_backward.end(), rule.x.begin(), [rule](T xi, T x_backward) { return abs(x_backward - (rule.b - xi)) < 1e-9; }));
}

TEST_CASE("gauss.cpp") {
    SECTION("Rule"){
        std::vector<DDouble> x(20), w(20);
        DDouble a = -1, b = 1;
        xprec::gauss_legendre(20, x.data(), w.data());
        Rule<DDouble> r1 = Rule<DDouble>(x, w);
        Rule<DDouble> r2 = Rule<DDouble>(x, w, a, b);
        REQUIRE(r1.a == r2.a);
        REQUIRE(r1.b == r2.b);
        REQUIRE(r1.x == r2.x);
        REQUIRE(r1.w == r2.w);
    }

    SECTION("legvander6"){
        int n = 6;
        auto result = legvander(legendre(n).x, n-1);
        Matrix<DDouble, Dynamic, Dynamic> expected(n, n);
        // expected is computed by
        // using SparseIR; m = SparseIR.legvander(SparseIR.legendre(6, SparseIR.Float64x2).x, 5); foreach(x -> println(x, ","), vec(m'))
        expected << 1.0,
                    -0.9324695142031520278123015544939835,
                    0.8042490923773935119886600608277198,
                    -0.6282499246436887457708844976782951,
                    0.4220050092706226656844451152082432,
                    -0.2057123110596225258297870187140517,
                    1.0,
                    -0.6612093864662645136613995950198845,
                    0.1557967791266409127010318094210897,
                    0.26911576974459911112396181357848563,
                    -0.428245862097120739542522563281485,
                    0.2943957149254374170467243373494173,
                    1.0,
                    -0.2386191860831969086305017216807169,
                    -0.4145913260494889701442373247943369,
                    0.3239618653539352481441754340495602,
                    0.1756623404298037786973215350969389,
                    -0.33461902074104083146186699361445111,
                    1.0,
                    0.2386191860831969086305017216807169,
                    -0.4145913260494889701442373247943369,
                    -0.3239618653539352481441754340495602,
                    0.1756623404298037786973215350969389,
                    0.33461902074104083146186699361445111,
                    1.0,
                    0.6612093864662645136613995950198845,
                    0.1557967791266409127010318094210897,
                    -0.26911576974459911112396181357848563,
                    -0.428245862097120739542522563281485,
                    -0.2943957149254374170467243373494173,
                    1.0,
                    0.9324695142031520278123015544939835,
                    0.8042490923773935119886600608277198,
                    0.6282499246436887457708844976782951,
                    0.4220050092706226656844451152082432,
                    0.2057123110596225258297870187140517;
        DDouble e = 1e-13;
        REQUIRE(result.isApprox(expected, e));
    }

    SECTION("legendre_collocation"){
        Rule<DDouble> r = legendre(2);
        Matrix<DDouble, Dynamic, Dynamic> result = legendre_collocation(r);
        Matrix<DDouble, Dynamic, Dynamic> expected(2, 2);
        // expected is computed by
        // julia> using SparseIR; m = SparseIR.legendre_collocation(SparseIR.legendre(2, SparseIR.Float64x2))
        expected << 0.5, 0.5,
                   -0.8660254037844386467637231707528938, 0.8660254037844386467637231707528938;
        DDouble e = 1e-13;
        REQUIRE(result.isApprox(expected, e));
    }

    SECTION("collocate") {
        int n = 6;
        Rule<DDouble> r = legendre(n);

        Eigen::Matrix<DDouble, Dynamic, Dynamic> cmat = legendre_collocation(r); // Assuming legendre_collocation function is defined

        Eigen::Matrix<DDouble, Dynamic, Dynamic> emat = legvander(r.x, r.x.size() - 1);
        DDouble e = 1e-13;
        Eigen::Matrix<DDouble, Dynamic, Dynamic> out =  emat * cmat;
        REQUIRE((emat * cmat).isApprox(Eigen::Matrix<DDouble, Dynamic, Dynamic>::Identity(n, n), e));
    }

    SECTION("gauss legendre") {
        int n = 200;
        Rule<DDouble> rule = legendre(n); // Assuming legendre function is defined
        gaussValidate(rule);
        std::vector<DDouble> x(n), w(n);
        xprec::gauss_legendre(n, x.data(), w.data());

        REQUIRE(rule.x == x);
        REQUIRE(rule.w == w);
    }

    SECTION("piecewise") {
        std::vector<DDouble> edges = {-4, -1, 1, 3};
        Rule<DDouble> rule = legendre(20).piecewise(edges);
        gaussValidate(rule);
    }
}