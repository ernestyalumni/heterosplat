# RasterizeToPixels3DGS oracle fixture

Captured by `Scripts/CaptureGsplatOracle.py` with `seed=42`,
gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.

I=1, N=8, 32x32, tile_size=16.
Pipeline: fully_fused_projection -> isect_tiles(sort=True) ->
isect_offset_encode -> rasterize_to_pixels.

- `means2d`       [1, 8, 2]
- `conics`        [1, 8, 3]
- `colors`        [1, 8, 3]
- `opacities`     [1, 8]
- `tile_offsets`  [1, 2, 2]
- `flatten_ids`   [8]
- `render_colors` [1, 32, 32, 3]
- `render_alphas` [1, 32, 32, 1]
