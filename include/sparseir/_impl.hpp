#pragma once

#include "xprec/ddouble.hpp"


namespace sparseir {
// General case
template<typename T>
T sqrt_impl(const T& x) {
    return std::sqrt(x);
}

// Specialization for DDouble
template<>
xprec::DDouble sqrt_impl(const xprec::DDouble& x) {
    return xprec::sqrt(x);
}

template<typename T >
T cosh_impl(const T& x) {
    return std::cosh(x);
}

template<>
xprec::DDouble cosh_impl(const xprec::DDouble& x) {
    return xprec::cosh(x);
}

template<typename T>
T sinh_impl(const T& x) {
    return std::sinh(x);
}

template<>
xprec::DDouble sinh_impl(const xprec::DDouble& x) {
    return xprec::sinh(x);
}

template<typename T>
T exp_impl(const T& x) {
    return std::exp(x);
}

template<>
xprec::DDouble exp_impl(const xprec::DDouble& x) {
    return xprec::exp(x);
}

}