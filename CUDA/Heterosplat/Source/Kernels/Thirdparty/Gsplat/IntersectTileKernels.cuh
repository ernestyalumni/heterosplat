/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from gsplat's cuda/csrc/IntersectTile.cu at
 * 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.
 *
 * heterosplat modification: extracted only the AccuTile/SNUGBOX device
 * helpers and templated __global__ intersect_tile_kernel. The ATen launcher,
 * CUB sorting helpers, and intersect_offset kernel are intentionally dropped.
 */

#pragma once

#include "Common.h"

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat
{

namespace cg = cooperative_groups;

__device__ inline float2 accutile_ellipse_intersection(
    float A, float B, float C, float disc, float t, float2 p,
    bool isY, float coord
) {
    float p_u   = isY ? p.y : p.x;
    float p_v   = isY ? p.x : p.y;
    float coeff = isY ? A : C;

    float h         = coord - p_u;
    float sqrt_term = sqrtf(disc * h * h + t * coeff);

    return {(-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v};
}

__device__ inline uint32_t accutile_process_tiles(
    float A, float B, float C, float disc, float t, float2 p,
    float2 bbox_min, float2 bbox_max, float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    uint32_t tile_size, uint32_t tile_width, bool isY,
    int64_t iid_enc, uint32_t tile_n_bits, int64_t depth_id_enc,
    uint32_t flatten_idx, int64_t *isect_ids, int32_t *flatten_ids,
    int64_t *cur_idx
) {
    float BLOCK = (float)tile_size;

    if (isY) {
        rect_min    = {rect_min.y, rect_min.x};
        rect_max    = {rect_max.y, rect_max.x};
        bbox_min    = {bbox_min.y, bbox_min.x};
        bbox_max    = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK;
    if (bbox_min.x <= min_line) {
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    } else {
        intersect_min_line = intersect_max_line;
    }

#pragma unroll 1
    for (int u = rect_min.x; u < rect_max.x; ++u) {
        max_line = min_line + BLOCK;
        if (max_line <= bbox_max.x) {
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
        }

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)));
        int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1)));

        tiles_count += max_tile_v - min_tile_v;

        if (isect_ids != nullptr) {
#pragma unroll 1
            for (int v = min_tile_v; v < max_tile_v; v++) {
                int64_t tile_id       = isY ? (int64_t)(u * tile_width + v) : (int64_t)(v * tile_width + u);
                isect_ids[*cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[*cur_idx] = static_cast<int32_t>(flatten_idx);
                ++(*cur_idx);
            }
        }

        intersect_min_line = intersect_max_line;
        min_line           = max_line;
    }
    return tiles_count;
}

template <typename scalar_t>
__global__ void intersect_tile_kernel(
    const bool packed,
    const uint32_t I,
    const uint32_t N,
    const uint32_t nnz,
    const int64_t *__restrict__ image_ids,
    const int64_t *__restrict__ gaussian_ids,
    const scalar_t *__restrict__ means2d,
    const int32_t *__restrict__ radii,
    const scalar_t *__restrict__ depths,
    const float *__restrict__ conics,
    const float *__restrict__ opacities,
    const int64_t *__restrict__ cum_tiles_per_gauss,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    int32_t *__restrict__ tiles_per_gauss,
    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids
) {
    (void)gaussian_ids;
    (void)image_n_bits;

    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : I * N)) {
        return;
    }

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    float2 mean2d = {(float)means2d[2 * idx], (float)means2d[2 * idx + 1]};

    int64_t iid_enc      = 0;
    int64_t depth_id_enc = 0;
    if (!first_pass) {
        int64_t iid;
        if (packed) {
            iid = image_ids[idx];
        } else {
            iid = idx / N;
        }
        iid_enc = iid << (32 + tile_n_bits);

        int32_t depth_i32 = *(int32_t *)&(depths[idx]);
        depth_id_enc = static_cast<uint32_t>(depth_i32);
    }

    if (conics != nullptr && opacities != nullptr) {
        const float A = conics[idx * 3];
        const float B = conics[idx * 3 + 1];
        const float C = conics[idx * 3 + 2];

        float disc = B * B - A * C;
        const float opacity = opacities[idx];
        float t = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity / ALPHA_THRESHOLD));

        float neg_t_over_disc = -t / disc;
        float x_extent = sqrtf(neg_t_over_disc * C);
        float y_extent = sqrtf(neg_t_over_disc * A);

        float2 bbox_min = {mean2d.x - x_extent, mean2d.y - y_extent};
        float2 bbox_max = {mean2d.x + x_extent, mean2d.y + y_extent};

        float Bx_over_C    = B * x_extent / C;
        float By_over_A    = B * y_extent / A;
        float2 bbox_argmin = {mean2d.y + Bx_over_C, mean2d.x + By_over_A};
        float2 bbox_argmax = {mean2d.y - Bx_over_C, mean2d.x - By_over_A};

        float tile_size_f = (float)tile_size;
        int2 rect_min = {max(0, min((int)tile_width,  (int)(bbox_min.x / tile_size_f))),
                         max(0, min((int)tile_height, (int)(bbox_min.y / tile_size_f)))};
        int2 rect_max = {max(0, min((int)tile_width,  (int)(bbox_max.x / tile_size_f + 1.f))),
                         max(0, min((int)tile_height, (int)(bbox_max.y / tile_size_f + 1.f)))};

        int y_span = rect_max.y - rect_min.y;
        int x_span = rect_max.x - rect_min.x;
        if (y_span * x_span == 0) {
            if (first_pass) tiles_per_gauss[idx] = 0;
            return;
        }

        bool isY = y_span < x_span;
        int64_t cur_idx = first_pass ? 0 : ((idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1]);

        uint32_t count = accutile_process_tiles(
            A, B, C, disc, t, mean2d,
            bbox_min, bbox_max, bbox_argmin, bbox_argmax,
            rect_min, rect_max,
            tile_size, tile_width, isY,
            iid_enc, tile_n_bits, depth_id_enc, idx,
            first_pass ? nullptr : isect_ids,
            first_pass ? nullptr : flatten_ids,
            &cur_idx
        );

        if (first_pass) {
            tiles_per_gauss[idx] = static_cast<int32_t>(count);
        }
    } else {
        float tile_radius_x = radius_x / static_cast<float>(tile_size);
        float tile_radius_y = radius_y / static_cast<float>(tile_size);
        float tile_x = mean2d.x / static_cast<float>(tile_size);
        float tile_y = mean2d.y / static_cast<float>(tile_size);

        int2 tile_min, tile_max;
        tile_min.x = min(max(0, (int32_t)floor(tile_x - tile_radius_x)), tile_width);
        tile_min.y = min(max(0, (int32_t)floor(tile_y - tile_radius_y)), tile_height);
        tile_max.x = min(max(0, (int32_t)ceil(tile_x + tile_radius_x)), tile_width);
        tile_max.y = min(max(0, (int32_t)ceil(tile_y + tile_radius_y)), tile_height);

        if (first_pass) {
            tiles_per_gauss[idx] = static_cast<int32_t>(
                (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
            );
            return;
        }

        int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
        for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
            for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                int64_t tile_id = i * tile_width + j;
                isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
            }
        }
    }
}

} // namespace gsplat
