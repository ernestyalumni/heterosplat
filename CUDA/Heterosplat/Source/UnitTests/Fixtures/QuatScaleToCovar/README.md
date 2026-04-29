# QuatScaleToCovarPreci oracle fixture

Captured by `Scripts/CaptureGsplatOracle.py` with
`number_of_gaussians=64`, `seed=42`,
layout `triu=False`, gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.

All `.bin` files are raw little-endian float32 except `N.bin`
which is little-endian uint32. Shapes:

- `quats`     [N, 4]
- `scales`    [N, 3]
- `covars`    [N, 3, 3] row-major
- `precis`    [N, 3, 3] row-major
- `v_covars`  [N, 3, 3] row-major (random upstream grad)
- `v_precis`  [N, 3, 3] row-major (random upstream grad)
- `v_quats`   [N, 4]    backward output
- `v_scales`  [N, 3]    backward output
