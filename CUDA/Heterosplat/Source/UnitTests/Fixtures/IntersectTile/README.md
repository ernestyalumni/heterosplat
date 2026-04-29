# IntersectTile oracle fixture

Captured by `Scripts/CaptureGsplatOracle.py` with `seed=42`,
gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5,
and `sort=False`, `packed=False`, `conics=None`, `opacities=None`.

All float buffers are raw little-endian float32. Integer buffers
are raw little-endian int32 or int64 as implied by filename;
shape scalars are little-endian uint32. Shapes:

- `means2d`             [2, 6, 2]
- `radii`               [2, 6, 2]
- `depths`              [2, 6]
- `tiles_per_gauss`     [2, 6]
- `cum_tiles_per_gauss` [12]
- `isect_ids`           [n_isects]
- `flatten_ids`         [n_isects]
