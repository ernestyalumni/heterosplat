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
    quat_scale_to_covar_preci,
    spherical_harmonics,
)


def write_float_tensor(path: str, tensor: torch.Tensor) -> None:
    """Write a CUDA/CPU tensor as raw little-endian float32 to disk."""
    array = tensor.detach().cpu().contiguous().to(torch.float32).numpy()
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

    print("==> done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
