# ProjectionEWA3DGSFused oracle fixture

Captured by `Scripts/CaptureGsplatOracle.py` with `seed=42`,
gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.

Pinhole camera, quat+scale path (no covars), no opacities,
eps2d=0.3, near_plane=0.01, far_plane=10000000000.0.

- `means`    [1, 8, 3]
- `quats`    [1, 8, 4]
- `scales`   [1, 8, 3]
- `viewmats` [1, 1, 4, 4]
- `Ks`       [1, 1, 3, 3]
- `radii`    [1, 1, 8, 2]
- `means2d`  [1, 1, 8, 2]
- `depths`   [1, 1, 8]
- `conics`   [1, 1, 8, 3]
