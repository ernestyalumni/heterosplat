# SphericalHarmonics oracle fixture

Captured by `Scripts/CaptureGsplatOracle.py` with
`number_of_gaussians=64`, `sh_degree=3`,
`seed=42`, gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.

All `.bin` files are raw little-endian float32 except
`N.bin`, `K.bin`, `degrees_to_use.bin` which are little-endian
uint32. Shapes (K = (sh_degree + 1)^2):

- `dirs`      [N, 3]
- `coeffs`    [N, K, 3]
- `colors`    [N, 3]    forward output
- `v_colors`  [N, 3]    random upstream grad
- `v_coeffs`  [N, K, 3] backward output
- `v_dirs`    [N, 3]    backward output
