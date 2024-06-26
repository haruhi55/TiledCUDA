#pragma once
#include "cuda_utils.hpp"

#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace tiledcuda::cell {
using namespace cute;

template <const int m, const int n, typename TiledMma>
DEVICE auto get_acc(const TiledMma& tiled_mma) {
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<m>, Int<n>>{});
    clear(acc);

    return acc;
}
}  // namespace tiledcuda::cell
