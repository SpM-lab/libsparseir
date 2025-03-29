#pragma once
#include "version.h"

namespace sparseir {

// Forward defitions
//class PowerOfTwo;
//class ExDouble;
//class DDouble;

// Forward declarations for classes
class Statistics;
class Fermionic;  // Forward declaration as class
class Bosonic;    // Forward declaration as class

template<typename T> class MatsubaraFreq;
template<typename T> class PiecewiseLegendreFT;
template<typename T> class PiecewiseLegendreFTVector;

class AbstractKernel;
template <typename K> class ReducedKernel;
template<typename T> class SVEHintsLogistic;
template<typename T> class SVEHintsRegularizedBose;
template<typename T> class SVEHintsReduced;
template<typename T> class AbstractSVEHints;

class PiecewiseLegendrePoly;

} // namespace sparseir