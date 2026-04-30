# IntersectOffset oracle fixture

Captured by `Scripts/CaptureGsplatOracle.py` with `seed=42`,
gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.

The sorted `isect_ids` come from `isect_tiles(..., sort=True)`,
then `isect_offset_encode()` produces the offsets.

Integer buffers are raw little-endian int32 or int64;
shape scalars are little-endian uint32. Shapes:

- `isect_ids_sorted` [30]
- `offsets`          [2, 4, 5]
