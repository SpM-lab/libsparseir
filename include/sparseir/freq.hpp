#include <iostream>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <type_traits>

// Abstract base class for Statistics
class Statistics {
public:
    virtual ~Statistics() = default;
    virtual int zeta() const = 0;
    virtual bool allowed(int n) const = 0;
};

// Fermionic statistics
class Fermionic : public Statistics {
public:
    int zeta() const override { return 1; }
    bool allowed(int n) const override { return n % 2 != 0; }
};

// Bosonic statistics
class Bosonic : public Statistics {
public:
    int zeta() const override { return 0; }
    bool allowed(int n) const override { return n % 2 == 0; }
};

// Factory function for Statistics
Statistics* createStatistics(int zeta) {
    if (zeta == 1) {
        return new Fermionic();
    } else if (zeta == 0) {
        return new Bosonic();
    } else {
        throw std::domain_error("does not correspond to known statistics");
    }
}

// MatsubaraFreq template class
template <typename S>
class MatsubaraFreq {
public:
    int n;

    MatsubaraFreq(S stat, int n) : n(n) {
        if (!stat.allowed(n)) {
            throw std::domain_error("Frequency " + std::to_string(n) + "π/β is not allowed");
        }
    }

    MatsubaraFreq(int n) : n(n) {
        S stat;
        if (!stat.allowed(n)) {
            throw std::domain_error("Frequency " + std::to_string(n) + "π/β is not allowed");
        }
    }

    int getN() const { return n; }

    double value(double beta) const {
        return n * (M_PI / beta);
    }

    std::complex<double> valueim(double beta) const {
        return std::complex<double>(0, 1) * value(beta);
    }

    int zeta() const {
        S stat;
        return stat.zeta();
    }

    // Overloaded operators
    MatsubaraFreq operator+(const MatsubaraFreq& other) const {
        return MatsubaraFreq(n + other.n);
    }

    MatsubaraFreq operator-(const MatsubaraFreq& other) const {
        return MatsubaraFreq(n - other.n);
    }

    MatsubaraFreq operator+() const {
        return *this;
    }

    MatsubaraFreq operator-() const {
        return MatsubaraFreq(-n);
    }

    MatsubaraFreq operator*(int c) const {
        return MatsubaraFreq(n * c);
    }

    friend MatsubaraFreq operator*(int c, const MatsubaraFreq& freq) {
        return freq * c;
    }

    int sign() const {
        return (n > 0) - (n < 0);
    }

    bool isZero() const {
        return n == 0;
    }

    bool operator<(const MatsubaraFreq& other) const {
        return n < other.n;
    }

    friend std::ostream& operator<<(std::ostream& os, const MatsubaraFreq& freq) {
        if (freq.n == 0) {
            os << "0";
        } else if (freq.n == 1) {
            os << "π/β";
        } else if (freq.n == -1) {
            os << "-π/β";
        } else {
            os << freq.n << "π/β";
        }
        return os;
    }
};

// Type aliases for convenience
using BosonicFreq = MatsubaraFreq<Bosonic>;
using FermionicFreq = MatsubaraFreq<Fermionic>;

// Utility functions
template <typename S>
int Integer(const MatsubaraFreq<S>& freq) {
    return freq.getN();
}

template <typename S>
int Int(const MatsubaraFreq<S>& freq) {
    return freq.getN();
}

template <typename S>
double value(const MatsubaraFreq<S>& freq, double beta) {
    return freq.value(beta);
}

template <typename S>
std::complex<double> valueim(const MatsubaraFreq<S>& freq, double beta) {
    return freq.valueim(beta);
}

template <typename S>
int zeta(const MatsubaraFreq<S>& freq) {
    return freq.zeta();
}

auto pioverbeta = MatsubaraFreq<Fermionic()>(1);

/*
int main() {
    try {
        BosonicFreq bf(2);
        FermionicFreq ff(1);

        std::cout << "BosonicFreq: " << bf << std::endl;
        std::cout << "FermionicFreq: " << ff << std::endl;

        std::cout << "Value of BosonicFreq: " << value(bf, 1.0) << std::endl;
        std::cout << "Value of FermionicFreq: " << value(ff, 1.0) << std::endl;

        std::cout << "Complex value of BosonicFreq: " << valueim(bf, 1.0) << std::endl;
        std::cout << "Complex value of FermionicFreq: " << valueim(ff, 1.0) << std::endl;

        std::cout << "Zeta of BosonicFreq: " << zeta(bf) << std::endl;
        std::cout << "Zeta of FermionicFreq: " << zeta(ff) << std::endl;

        auto sum = bf + bf;
        std::cout << "Sum of BosonicFreq: " << sum << std::endl;

        auto diff = ff - bf;
        std::cout << "Difference of FermionicFreq and BosonicFreq: " << diff << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
*/