#pragma once
#include "version.h"

namespace sparseir {

// Forward defitions
//class PowerOfTwo;
//class ExDouble;
//class DDouble;

// Forward declarations for classes
class Statistics;
class Fermionic;  // クラスとして前方宣言
class Bosonic;    // クラスとして前方宣言

template <typename K> class ReducedKernel;
template<typename T> class SVEHintsLogistic;
template<typename T> class SVEHintsRegularizedBose;
template<typename T> class SVEHintsReduced;

} // namespace sparseir