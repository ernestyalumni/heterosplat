/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from gsplat's cuda/csrc/IntersectTile.cu at
 * 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.
 *
 * heterosplat modification: extracted only the __global__
 * intersect_offset_kernel. The ATen launcher is intentionally dropped.
 */

#pragma once

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat
{

namespace cg = cooperative_groups;

__global__ void intersect_offset_kernel(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [I, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t iid_curr = isect_id_curr >> (tile_n_bits);
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = iid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < I * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (bid, cid,
        // tile_id) tuple changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t iid_prev = isect_id_prev >> (tile_n_bits);
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = iid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

} // namespace gsplat
