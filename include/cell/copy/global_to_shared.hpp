#pragma once

#include "cell/copy/constants.hpp"
#include "cell/copy/dyn_copy.hpp"
#include "cell/copy/warp.hpp"
#include "cell/traits/base.hpp"
#include "types/mod.hpp"

namespace tiledcuda::cell::copy {
using namespace traits;
namespace tl = tile_layout;

namespace {
template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
DEVICE void copy_base_tile_g2s(const Element* src_data, Element* dst_data,
                               SrcLayout src_layout, DstLayout dst_layout,
                               TiledCopy tiled_copy) {
    int tid = threadIdx.x;

    auto gtile = make_tensor(make_gmem_ptr(src_data), src_layout);
    auto stile = make_tensor(make_smem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(gtile);
    auto dst = loader.partition_D(stile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}
}  // namespace

template <typename Global, typename Shared, const int kRowExec,
          const int kColExec, typename WarpLayout, const tl::Layout kType>
struct GlobalToSharedLoaderImpl {
    using DType = Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, const int kRowExec,
          const int kColExec, typename WarpLayout_>
struct GlobalToSharedLoaderImpl<Global_, Shared_, kRowExec, kColExec,
                                WarpLayout, tl::Layout::kRowMajor> {
    using Global = Global_;
    using Shared = Shared_;
    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");

    using DType = Global::DType;

    using WarpLayout = WarpLayout_;

    // In the row-major layout, columns are contiguous in memory. 2 threads in a
    // warp access data along the continguous dimension (columns), and 8 warps
    // access data along the strided dimension (rows).
    using WarpThreadLayout = tl::RowMajor<16, 2>;
    static constexpr int kThreadRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;
    // Efficient memory access requires specifying the tile shape.
    static_assert(Global::kCols % (kThreadCols * kNumPerAccess) == 0,
                  "The columns of a GlobalTile should be divisible by the "
                  "number of threads per row in the thread layout.");
    static_assert(Global::kRows % kThreadRows == 0,
                  "The rows of a GlobalTile should be divisible by the "
                  "number of threads per column in the thread layout.");

#ifdef CP_ASYNC_SM80_ENABLED
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DType>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, DType>;
#endif

    using BaseShape = traits::BaseTileShape<DType>;

    // using GlobalLayout = Global::Layout::CuteLayout;
    // using SharedLayout = Shared::Layout::CuteLayout;
    // using TiledCopy = decltype(make_tiled_copy(
    //     CopyInst{},
    //     cute::Layout<Shape<Int<kThreadRows>, Int<kThreadCols>>,
    //                  Stride<Int<kThreadCols>, _1>>{},
    //     cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{},
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<Baseshape::kCols>>,
                     Stride<Int<BaseShape::kCols>, _1>>{},
        cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    // DEVICE GlobalToSharedLoaderImpl()
    //     : src_layout_(GlobalLayout{}),
    //       dst_layout_(SharedLayout{}),
    //       tiled_copy_(TiledCopy{}) {}

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_base_tile_g2s(src, dst, src_layout_, dst_layout_, tiled_copy_);
    }

    //   private:
    //     GlobalLayout src_layout_;
    //     SharedLayout dst_layout_;
    //     TiledCopy tiled_copy_;
};

template <typename Global, typename Shared, typename WarpLayout,
          const tl::Layout kType>
struct SharedToGlobalStorerImpl {
    using DType = Global::DType;

    DEVICE void operator()(const DType* src, DType* dst);
};

template <typename Global_, typename Shared_, typename WarpLayout_>
struct SharedToGlobalStorerImpl<Global_, Shared_, WarpLayout_,
                                tl::Layout::kRowMajor> {
    using Shared = Shared_;
    using Global = Global_;
    static_assert(Global::kRows == Shared::kRows &&
                      Global::kCols == Shared::kCols,
                  "Global and shared memory should have the same shape.");

    using DType = Global::DType;

    using WarpLayout = WarpLayout_;
    // TODO: configuration for how threads are laid out in a single warp,
    // hard-coded it for now. Make it configurable in future implementation.
    // In the row-major layout, columns are contiguous in memory. 4 threads in a
    // warp access data along the continguous dimension (columns), and 8 warps
    // access data along the strided dimension (rows).
    using WarpThreadLayout = tl::RowMajor<16, 2>;
    static constexpr int kThreadRows =
        tl::num_rows<WarpLayout> * tl::num_rows<WarpThreadLayout>;
    static constexpr int kThreadCols =
        tl::num_cols<WarpLayout> * tl::num_cols<WarpThreadLayout>;

    static constexpr int kNumPerAccess = TraitsBase<DType>::kNumPerAccess;
    // Efficient memory access requires specifying the tile shape.
    static_assert(Global::kCols % (kThreadCols * kNumPerAccess) == 0,
                  "The columns of a GlobalTile should be divisible by the "
                  "number of threads per row in the thread layout.");
    static_assert(Global::kRows % kThreadRows == 0,
                  "The rows of a GlobalTile should be divisible by the "
                  "number of threads per column in the thread layout.");

    // transfer data from global memory to shared memory has cp.async,
    // while transfer data from shared memory to global memory does not
    // have. For the latter case, the copy instruction should be the
    // default one.
    using BaseShape = traits::BaseTileShape<DType>;

    using GlobalLayout = Global::Layout::CuteLayout;
    using SharedLayout = Shared::Layout::CuteLayout;

    // copy a 16x16 BaseTile, using CuTe's tiled_copy
    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, DType>{},
        cute::Layout<Shape<Int<BaseShape::kRows>, Int<BaseShape::kCols>>,
                     Stride<Int<BaseShape::kCols>, _1>>{},
        cute::Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    DEVICE SharedToGlobalStorerImpl()
        : src_layout_(SharedLayout{}),
          dst_layout_(GlobalLayout{}),
          tiled_copy_(TiledCopy{}) {}

    DEVICE void operator()(const DType* src, DType* dst) {
        copy_2d_tile_s2g(src, dst, src_layout_, dst_layout_, tiled_copy_);
    }

  private:
    SharedLayout src_layout_;
    GlobalLayout dst_layout_;
    TiledCopy tiled_copy_;
};

template <typename Shared_, typename WarpLayout_,
          const tl::Layout kType_ = tl::Layout::kRowMajor>
struct GlobalToSharedLoader {
    using Shared = Shared_;
    using DType = Shared::DType;
    using WarpLayout = WarpLayout_;

    static constexpr tl::Layout kType = kType_;
    using BaseShape = BaseTileShape<DType>;

    static constexpr int kRowExec = Shared::Layout::kRows / BaseShape::kRows;
    static constexpr int kColExec = Shared::Layout::kCols / BaseShape::kCols;

    template <typename Global>
    DEVICE void operator()(const Global& src, Shared& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Loader =
            GlobalToSharedLoaderImpl<Global, Shared, kRowExec, kColExec, kType>;

        Loader loader;
        loader(src_ptr, dst_ptr);
    }
};

template <typename Shared_, typename WarpLayout_,
          const tl::Layout kType_ = tl::Layout::kRowMajor>
struct SharedToGlobalStorer {
    using Shared = Shared_;
    using DType = Shared::DType;
    using WarpLayout = WarpLayout_;

    static constexpr tl::Layout kType = kType_;

    template <typename Global>
    DEVICE void operator()(const Shared& src, Global& dst) {
        const DType* src_ptr = src.data();
        DType* dst_ptr = dst.mutable_data();

        using Storer =
            SharedToGlobalStorerImpl<Global, Shared, WarpLayout, kType>;

        Storer storer;
        storer(src_ptr, dst_ptr);
    }
};

}  // namespace tiledcuda::cell::copy
