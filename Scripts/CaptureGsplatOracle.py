"""Capture gsplat-Python forward+backward outputs as fixtures.

heterosplat's torch-free C++ launchers (Source/Kernels/Heterosplat/) call
exactly the same vendored CUDA __global__ kernels as gsplat-Python. Running
gsplat-Python on seeded random inputs and dumping inputs+outputs to disk
gives us a numerical-correctness oracle that the C++ tests can compare
against bit-for-bit (or to ~1e-5 since float32 roundoff in the kernels is
deterministic but accumulated reductions vary at the ULP level).

The script must run inside the heterosplat:26.02-py3 container (where
torch + the gsplat extension are pre-installed). Invocation:

    ./Scripts/run_container.sh \\
      'python3 /heterosplat/Scripts/CaptureGsplatOracle.py /heterosplat/CUDA/Heterosplat/Source/UnitTests/Fixtures'

Each captured fixture set lives in its own subdirectory under the output
root. All numerical buffers are written as raw little-endian float32; all
shape scalars (N, K, degree) as raw little-endian uint32. The fixture C++
loader (Source/UnitTests/OracleFixture.h) reads them back with `std::ifstream`
and `read()` -- no header parsing, no dependency on a binary format library.
"""

import argparse
import os
import struct
import sys

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    quat_scale_to_covar_preci,
    spherical_harmonics,
)


def write_float_tensor(path: str, tensor: torch.Tensor) -> None:
    """Write a CUDA/CPU tensor as raw little-endian float32 to disk."""
    array = tensor.detach().cpu().contiguous().to(torch.float32).numpy()
    with open(path, "wb") as f:
        f.write(array.tobytes())


def write_int32_tensor(path: str, tensor: torch.Tensor) -> None:
    """Write a CUDA/CPU tensor as raw little-endian int32 to disk."""
    array = tensor.detach().cpu().contiguous().to(torch.int32).numpy()
    with open(path, "wb") as f:
        f.write(array.tobytes())


def write_int64_tensor(path: str, tensor: torch.Tensor) -> None:
    """Write a CUDA/CPU tensor as raw little-endian int64 to disk."""
    array = tensor.detach().cpu().contiguous().to(torch.int64).numpy()
    with open(path, "wb") as f:
        f.write(array.tobytes())


def write_uint32(path: str, value: int) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", int(value)))


def capture_quat_scale_to_covar(out_dir: str, number_of_gaussians: int = 64,
                                 seed: int = 42) -> None:
    """Forward + backward fixture for quat_scale_to_covar_preci, full layout."""
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Inputs: random quats (un-normalised; the kernel normalises in-line);
    # positive scales bounded away from zero to keep the precision branch
    # numerically well-conditioned.
    quats = torch.randn(
        number_of_gaussians, 4, generator=generator,
        device="cuda", dtype=torch.float32)
    scales = (
        torch.rand(
            number_of_gaussians, 3, generator=generator,
            device="cuda", dtype=torch.float32) * 0.5 + 0.1)

    quats_grad = quats.clone().requires_grad_(True)
    scales_grad = scales.clone().requires_grad_(True)

    covars, precis = quat_scale_to_covar_preci(
        quats_grad, scales_grad,
        compute_covar=True, compute_preci=True, triu=False)

    # Random upstream gradients for the backward pass.
    v_covars = torch.randn(
        number_of_gaussians, 3, 3, generator=generator,
        device="cuda", dtype=torch.float32)
    v_precis = torch.randn(
        number_of_gaussians, 3, 3, generator=generator,
        device="cuda", dtype=torch.float32)

    # Backward through both branches simultaneously: scalar loss is the inner
    # product of upstream grad with output, summed over both branches.
    loss = (covars * v_covars).sum() + (precis * v_precis).sum()
    loss.backward()

    v_quats = quats_grad.grad.detach()
    v_scales = scales_grad.grad.detach()

    write_uint32(os.path.join(out_dir, "N.bin"), number_of_gaussians)
    write_float_tensor(os.path.join(out_dir, "quats.bin"), quats)
    write_float_tensor(os.path.join(out_dir, "scales.bin"), scales)
    write_float_tensor(os.path.join(out_dir, "covars.bin"), covars)
    write_float_tensor(os.path.join(out_dir, "precis.bin"), precis)
    write_float_tensor(os.path.join(out_dir, "v_covars.bin"), v_covars)
    write_float_tensor(os.path.join(out_dir, "v_precis.bin"), v_precis)
    write_float_tensor(os.path.join(out_dir, "v_quats.bin"), v_quats)
    write_float_tensor(os.path.join(out_dir, "v_scales.bin"), v_scales)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "# QuatScaleToCovarPreci oracle fixture\n\n"
            f"Captured by `Scripts/CaptureGsplatOracle.py` with\n"
            f"`number_of_gaussians={number_of_gaussians}`, `seed={seed}`,\n"
            f"layout `triu=False`, gsplat upstream commit "
            "53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.\n\n"
            "All `.bin` files are raw little-endian float32 except `N.bin`\n"
            "which is little-endian uint32. Shapes:\n\n"
            "- `quats`     [N, 4]\n"
            "- `scales`    [N, 3]\n"
            "- `covars`    [N, 3, 3] row-major\n"
            "- `precis`    [N, 3, 3] row-major\n"
            "- `v_covars`  [N, 3, 3] row-major (random upstream grad)\n"
            "- `v_precis`  [N, 3, 3] row-major (random upstream grad)\n"
            "- `v_quats`   [N, 4]    backward output\n"
            "- `v_scales`  [N, 3]    backward output\n")

    print(f"  wrote QuatScaleToCovar fixture: N={number_of_gaussians} -> {out_dir}")


def capture_spherical_harmonics(out_dir: str, number_of_gaussians: int = 64,
                                 sh_degree: int = 3, seed: int = 42) -> None:
    """Forward + backward fixture for spherical_harmonics."""
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    coefficients_per_gaussian = (sh_degree + 1) ** 2  # 1, 4, 9, 16, 25 for L = 0..4

    dirs = torch.randn(
        number_of_gaussians, 3, generator=generator,
        device="cuda", dtype=torch.float32)
    coeffs = torch.randn(
        number_of_gaussians, coefficients_per_gaussian, 3, generator=generator,
        device="cuda", dtype=torch.float32)

    dirs_grad = dirs.clone().requires_grad_(True)
    coeffs_grad = coeffs.clone().requires_grad_(True)

    colors = spherical_harmonics(sh_degree, dirs_grad, coeffs_grad, masks=None)

    v_colors = torch.randn(
        number_of_gaussians, 3, generator=generator,
        device="cuda", dtype=torch.float32)
    loss = (colors * v_colors).sum()
    loss.backward()

    v_dirs = dirs_grad.grad.detach()
    v_coeffs = coeffs_grad.grad.detach()

    write_uint32(os.path.join(out_dir, "N.bin"), number_of_gaussians)
    write_uint32(os.path.join(out_dir, "K.bin"), coefficients_per_gaussian)
    write_uint32(os.path.join(out_dir, "degrees_to_use.bin"), sh_degree)
    write_float_tensor(os.path.join(out_dir, "dirs.bin"), dirs)
    write_float_tensor(os.path.join(out_dir, "coeffs.bin"), coeffs)
    write_float_tensor(os.path.join(out_dir, "colors.bin"), colors)
    write_float_tensor(os.path.join(out_dir, "v_colors.bin"), v_colors)
    write_float_tensor(os.path.join(out_dir, "v_coeffs.bin"), v_coeffs)
    write_float_tensor(os.path.join(out_dir, "v_dirs.bin"), v_dirs)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "# SphericalHarmonics oracle fixture\n\n"
            f"Captured by `Scripts/CaptureGsplatOracle.py` with\n"
            f"`number_of_gaussians={number_of_gaussians}`, `sh_degree={sh_degree}`,\n"
            f"`seed={seed}`, gsplat upstream commit "
            "53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.\n\n"
            "All `.bin` files are raw little-endian float32 except\n"
            "`N.bin`, `K.bin`, `degrees_to_use.bin` which are little-endian\n"
            "uint32. Shapes (K = (sh_degree + 1)^2):\n\n"
            "- `dirs`      [N, 3]\n"
            "- `coeffs`    [N, K, 3]\n"
            "- `colors`    [N, 3]    forward output\n"
            "- `v_colors`  [N, 3]    random upstream grad\n"
            "- `v_coeffs`  [N, K, 3] backward output\n"
            "- `v_dirs`    [N, 3]    backward output\n")

    print(f"  wrote SphericalHarmonics fixture: N={number_of_gaussians} "
          f"L={sh_degree} -> {out_dir}")


def capture_intersect_tile(out_dir: str, seed: int = 42) -> None:
    """Forward fixture for intersect_tile with unsorted raw-kernel output."""
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    number_of_images = 2
    number_of_gaussians = 6
    tile_size = 8
    tile_width = 5
    tile_height = 4

    means2d = torch.rand(
        number_of_images, number_of_gaussians, 2,
        generator=generator, device="cuda", dtype=torch.float32)
    means2d[..., 0] *= tile_width * tile_size
    means2d[..., 1] *= tile_height * tile_size

    radii = torch.randint(
        1, 8, (number_of_images, number_of_gaussians, 2),
        generator=generator, device="cuda", dtype=torch.int32)
    # Exercise the zero-radius cull path.
    radii[0, 1, 0] = 0

    depths = (
        torch.rand(
            number_of_images, number_of_gaussians,
            generator=generator, device="cuda", dtype=torch.float32) * 4.0
        + 0.25)

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=False,
        segmented=False,
        packed=False,
        conics=None,
        opacities=None)
    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.reshape(-1), 0)

    write_uint32(os.path.join(out_dir, "I.bin"), number_of_images)
    write_uint32(os.path.join(out_dir, "N.bin"), number_of_gaussians)
    write_uint32(os.path.join(out_dir, "tile_size.bin"), tile_size)
    write_uint32(os.path.join(out_dir, "tile_width.bin"), tile_width)
    write_uint32(os.path.join(out_dir, "tile_height.bin"), tile_height)
    write_uint32(os.path.join(out_dir, "n_isects.bin"), isect_ids.numel())
    write_float_tensor(os.path.join(out_dir, "means2d.bin"), means2d)
    write_int32_tensor(os.path.join(out_dir, "radii.bin"), radii)
    write_float_tensor(os.path.join(out_dir, "depths.bin"), depths)
    write_int32_tensor(os.path.join(out_dir, "tiles_per_gauss.bin"), tiles_per_gauss)
    write_int64_tensor(os.path.join(out_dir, "cum_tiles_per_gauss.bin"), cum_tiles_per_gauss)
    write_int64_tensor(os.path.join(out_dir, "isect_ids.bin"), isect_ids)
    write_int32_tensor(os.path.join(out_dir, "flatten_ids.bin"), flatten_ids)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "# IntersectTile oracle fixture\n\n"
            f"Captured by `Scripts/CaptureGsplatOracle.py` with `seed={seed}`,\n"
            "gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5,\n"
            "and `sort=False`, `packed=False`, `conics=None`, `opacities=None`.\n\n"
            "All float buffers are raw little-endian float32. Integer buffers\n"
            "are raw little-endian int32 or int64 as implied by filename;\n"
            "shape scalars are little-endian uint32. Shapes:\n\n"
            f"- `means2d`             [{number_of_images}, {number_of_gaussians}, 2]\n"
            f"- `radii`               [{number_of_images}, {number_of_gaussians}, 2]\n"
            f"- `depths`              [{number_of_images}, {number_of_gaussians}]\n"
            f"- `tiles_per_gauss`     [{number_of_images}, {number_of_gaussians}]\n"
            f"- `cum_tiles_per_gauss` [{number_of_images * number_of_gaussians}]\n"
            "- `isect_ids`           [n_isects]\n"
            "- `flatten_ids`         [n_isects]\n")

    print(f"  wrote IntersectTile fixture: I={number_of_images} "
          f"N={number_of_gaussians} n_isects={isect_ids.numel()} -> {out_dir}")


def capture_intersect_offset(out_dir: str, seed: int = 42) -> None:
    """Forward fixture for intersect_offset using sorted isect_ids from isect_tiles."""
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    number_of_images = 2
    number_of_gaussians = 6
    tile_size = 8
    tile_width = 5
    tile_height = 4

    means2d = torch.rand(
        number_of_images, number_of_gaussians, 2,
        generator=generator, device="cuda", dtype=torch.float32)
    means2d[..., 0] *= tile_width * tile_size
    means2d[..., 1] *= tile_height * tile_size

    radii = torch.randint(
        1, 8, (number_of_images, number_of_gaussians, 2),
        generator=generator, device="cuda", dtype=torch.int32)
    radii[0, 1, 0] = 0

    depths = (
        torch.rand(
            number_of_images, number_of_gaussians,
            generator=generator, device="cuda", dtype=torch.float32) * 4.0
        + 0.25)

    _tiles_per_gauss, isect_ids_sorted, _flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
        packed=False,
        conics=None,
        opacities=None)

    offsets = isect_offset_encode(
        isect_ids_sorted, number_of_images, tile_width, tile_height)

    n_isects = isect_ids_sorted.numel()

    write_uint32(os.path.join(out_dir, "I.bin"), number_of_images)
    write_uint32(os.path.join(out_dir, "n_tiles.bin"), tile_width * tile_height)
    write_uint32(os.path.join(out_dir, "tile_width.bin"), tile_width)
    write_uint32(os.path.join(out_dir, "tile_height.bin"), tile_height)
    write_uint32(os.path.join(out_dir, "n_isects.bin"), n_isects)
    write_int64_tensor(os.path.join(out_dir, "isect_ids_sorted.bin"),
                       isect_ids_sorted)
    write_int32_tensor(os.path.join(out_dir, "offsets.bin"), offsets)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "# IntersectOffset oracle fixture\n\n"
            f"Captured by `Scripts/CaptureGsplatOracle.py` with `seed={seed}`,\n"
            "gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.\n\n"
            "The sorted `isect_ids` come from `isect_tiles(..., sort=True)`,\n"
            "then `isect_offset_encode()` produces the offsets.\n\n"
            "Integer buffers are raw little-endian int32 or int64;\n"
            "shape scalars are little-endian uint32. Shapes:\n\n"
            f"- `isect_ids_sorted` [{n_isects}]\n"
            f"- `offsets`          [{number_of_images}, {tile_height}, {tile_width}]\n")

    print(f"  wrote IntersectOffset fixture: I={number_of_images} "
          f"n_isects={n_isects} -> {out_dir}")


def capture_projection_ewa_3dgs_fused(out_dir: str, seed: int = 42) -> None:
    """Forward + backward fixture for projection_ewa_3dgs_fused (pinhole, quat+scale path)."""
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    B = 1
    C = 1
    N = 8
    image_width = 128
    image_height = 128
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1e10

    means = torch.randn(B, N, 3, generator=generator, device="cuda",
                         dtype=torch.float32)
    means[..., 2] = means[..., 2].abs() + 1.0  # ensure positive depth

    quats = torch.randn(B, N, 4, generator=generator, device="cuda",
                         dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    scales = torch.rand(B, N, 3, generator=generator, device="cuda",
                         dtype=torch.float32) * 0.3 + 0.01

    viewmats = torch.eye(4, device="cuda", dtype=torch.float32).reshape(
        1, 1, 4, 4).expand(B, C, -1, -1).contiguous()

    Ks = torch.tensor(
        [[[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]]],
        device="cuda", dtype=torch.float32
    ).reshape(1, 1, 3, 3).expand(B, C, -1, -1).contiguous()

    # Forward (no autograd needed for oracle capture)
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars=None, quats=quats, scales=scales,
        viewmats=viewmats, Ks=Ks,
        width=image_width, height=image_height,
        eps2d=eps2d, near_plane=near_plane, far_plane=far_plane,
        radius_clip=0.0, packed=False, calc_compensations=False,
        camera_model="pinhole", opacities=None)

    write_uint32(os.path.join(out_dir, "B.bin"), B)
    write_uint32(os.path.join(out_dir, "C.bin"), C)
    write_uint32(os.path.join(out_dir, "N.bin"), N)
    write_uint32(os.path.join(out_dir, "image_width.bin"), image_width)
    write_uint32(os.path.join(out_dir, "image_height.bin"), image_height)
    write_float_tensor(os.path.join(out_dir, "means.bin"), means)
    write_float_tensor(os.path.join(out_dir, "quats.bin"), quats)
    write_float_tensor(os.path.join(out_dir, "scales.bin"), scales)
    write_float_tensor(os.path.join(out_dir, "viewmats.bin"), viewmats)
    write_float_tensor(os.path.join(out_dir, "Ks.bin"), Ks)
    write_int32_tensor(os.path.join(out_dir, "radii.bin"), radii)
    write_float_tensor(os.path.join(out_dir, "means2d.bin"), means2d)
    write_float_tensor(os.path.join(out_dir, "depths.bin"), depths)
    write_float_tensor(os.path.join(out_dir, "conics.bin"), conics)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "# ProjectionEWA3DGSFused oracle fixture\n\n"
            f"Captured by `Scripts/CaptureGsplatOracle.py` with `seed={seed}`,\n"
            "gsplat upstream commit 53f89aa58fdbe6bf1b442975e1e4b7d5411e94e5.\n\n"
            "Pinhole camera, quat+scale path (no covars), no opacities,\n"
            f"eps2d={eps2d}, near_plane={near_plane}, far_plane={far_plane}.\n\n"
            f"- `means`    [{B}, {N}, 3]\n"
            f"- `quats`    [{B}, {N}, 4]\n"
            f"- `scales`   [{B}, {N}, 3]\n"
            f"- `viewmats` [{B}, {C}, 4, 4]\n"
            f"- `Ks`       [{B}, {C}, 3, 3]\n"
            f"- `radii`    [{B}, {C}, {N}, 2]\n"
            f"- `means2d`  [{B}, {C}, {N}, 2]\n"
            f"- `depths`   [{B}, {C}, {N}]\n"
            f"- `conics`   [{B}, {C}, {N}, 3]\n")

    print(f"  wrote ProjectionEWA3DGSFused fixture: B={B} C={C} N={N} "
          f"-> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root",
                        help="Directory under which fixture subdirs are written")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- run inside the heterosplat container "
              "with --gpus.", file=sys.stderr)
        return 1

    print(f"==> torch={torch.__version__}, cuda={torch.version.cuda}, "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"==> writing fixtures under {args.output_root}")

    capture_quat_scale_to_covar(
        os.path.join(args.output_root, "QuatScaleToCovar"),
        seed=args.seed)
    capture_spherical_harmonics(
        os.path.join(args.output_root, "SphericalHarmonics"),
        seed=args.seed)
    capture_intersect_tile(
        os.path.join(args.output_root, "IntersectTile"),
        seed=args.seed)
    capture_intersect_offset(
        os.path.join(args.output_root, "IntersectOffset"),
        seed=args.seed)
    capture_projection_ewa_3dgs_fused(
        os.path.join(args.output_root, "ProjectionEWA3DGSFused"),
        seed=args.seed)

    print("==> done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
