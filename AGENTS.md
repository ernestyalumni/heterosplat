# AGENTS.md — heterosplat

heterosplat is a single-author project (Ernest Yeung) that fuses heterogeneous 3D sources into Gaussian splat scenes + meshes. **Built in pure C++ + CUDA C++** — no Python, no Rust, no PyTorch.

This file is the entry point for any AI agent (Claude, Codex, OpenClaw, …) working in this repo.

## Read first

1. **`PLAN.md`** — the source of truth for what's being built, why, and in what order. Phases, done-criteria, locked tech decisions, open questions, status table.
2. **`README.md`** — one-line thesis (kept terse until first public ship).

If you only have time for one file, read `PLAN.md`.

## Big picture

heterosplat is the primary **World Labs portfolio project**. The interview cycle is live (tech screen passed 2026-04-21). Each phase in `PLAN.md` ends with a public ship (X / blog post / GitHub README) — that's the cadence that makes this project's existence worth the time.

## Locked decisions (do not re-litigate)

These were debated, decided, and written down. If a future you wants to revisit, there has to be a new fact, not just a fresh opinion:

- **Pure C++17 + CUDA C++.** No Python, no PyTorch, no Rust. Reasoning lives in `PLAN.md` → "Why C++ + CUDA, no Python."
- **Vendor gsplat's `csrc/` kernels** (Apache-2.0); replace the torch-thin launchers with raw-pointer launchers we own. `gsplat`'s `__global__` kernels are torch-free; only the launchers + ATen layer touch `at::Tensor`.
- **Single CMake project, single static binary** `heterosplat` with subcommands (`train`, `render`, `view`, `normalize`).
- **Single-header dependencies** vendored in `thirdparty/`: GLM, stb, CLI11, nlohmann/json, fmt; GLFW + Dear ImGui + glad for the Phase 3 viewer. Tests use **GoogleTest via FetchContent** (matches `InServiceOfX/CUDALibraries/MoreCUDA`), single `Check` executable.
- **Linux-only** until Phase 4+.
- **License: Apache 2.0** (matches vendored gsplat). The current `LICENSE` is MIT — switch before first public ship.
- **"Heterogeneous" = exactly two source formats** across the whole project. COLMAP in Phase 1 + one of {mesh, depth-maps, second-coord-convention capture} in Phase 2. Anything more is Phase 4+.

## Where the work happens

- Repo: `/home/propdev/.openclaw/workspace/workspace2/repos/heterosplat/`
- Dev container: `heterosplat:26.02-py3` (built from `repos/InServiceOfX/Deployments/DockerContainers/Builds/Physics/Heterosplat/`)
- Vendored kernel source: `repos/gsplat/gsplat/cuda/{csrc,include}/`. Apache-2.0; preserve LICENSE + NOTICE.
- Source tree: `CUDA/Heterosplat/Source/` (mirrors the `InServiceOfX/CUDALibraries/MoreCUDA/Source/` layout).

```
heterosplat/
├── PLAN.md, AGENTS.md, README.md          # docs (this file)
├── Documents/LaTeX/KernelMathematics.tex  # math reference, one section per kernel
├── Scripts/
│   ├── run_container.sh                   # docker dev shell / one-shot
│   ├── run_tests.sh                       # host test runner (forwards args to ./Check)
│   ├── run_configuration.yml.example      # per-machine GPU id template (gitignored real)
│   └── CaptureGsplatOracle.py             # captures gsplat-Python fixtures (in container)
└── CUDA/Heterosplat/
    ├── Build/                             # local build dir (gitignored), `cmake ../Source`
    └── Source/
        ├── CMakeLists.txt                 # top-level
        ├── Core/{CMakeLists.txt, Tensor.h}
        ├── Kernels/
        │   ├── Thirdparty/Gsplat/         # vendored Apache-2.0 (LICENSE, NOTICE, VENDORED.md)
        │   │   ├── Common.h               # patched: torch macros stripped
        │   │   ├── Config.h, Utils.cuh    # verbatim
        │   │   ├── QuatScaleToCovarKernels.cuh
        │   │   ├── SphericalHarmonicsKernels.cuh   (gpuAtomicAdd → atomicAdd)
        │   │   ├── IntersectTileKernels.cuh
        │   │   ├── IntersectOffsetKernels.cuh
        │   │   └── ProjectionEWA3DGSFusedKernels.cuh   (gpuAtomicAdd → atomicAdd)
        │   └── Heterosplat/               # our raw-pointer launchers
        │       ├── IntersectOffset.{h,cu}
        │       ├── IntersectTile.{h,cu}
        │       ├── ProjectionEWA3DGSFused.{h,cu}
        │       ├── QuatScaleToCovar.{h,cu}
        │       └── SphericalHarmonics.{h,cu}
        └── UnitTests/                     # single Check executable, gtest_discover
            ├── DeviceBuffer.h             # typed GPU buffer for tests
            ├── OracleFixture.h            # tiny .bin loader
            ├── Fixtures/                  # captured gsplat-Python outputs
            │   ├── IntersectOffset/
            │   ├── IntersectTile/
            │   ├── ProjectionEWA3DGSFused/
            │   ├── QuatScaleToCovar/
            │   └── SphericalHarmonics/
            └── Kernels/Heterosplat/
                ├── *_tests.cu             # closed-form + gradcheck
                └── *_oracle_tests.cu      # vs gsplat-Python
```

## Build & test

### Per-machine setup (once)

`gpu_id` in `Scripts/run_configuration.yml` is **PCI-bus ordering** (matches `nvidia-smi -L` and docker `--gpus device=N`). On hosts where CUDA's default ordering differs (e.g. RTX 3060 + GTX 980 Ti), the host script forces `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the same value works in both contexts.

```bash
cp Scripts/run_configuration.yml.example Scripts/run_configuration.yml
$EDITOR Scripts/run_configuration.yml   # set gpu_id (PCI index of your sm_86+ GPU)
```

### Host build (the fast dev loop)

```bash
mkdir -p CUDA/Heterosplat/Build && cd CUDA/Heterosplat/Build
cmake ../Source && make -j6
cd -                       # back to repo root
./Scripts/run_tests.sh                                # all tests
./Scripts/run_tests.sh --gtest_filter='QuatScaleToCovar.*'   # filter
```

The script reads `gpu_id` from the yml, exports `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=$gpu_id`, then execs `./build/Check`. Forwards extra args to gtest. Manual one-liner equivalent: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./CUDA/Heterosplat/Build/Check`.

### Container build (oracle capture, deploy-shape smoke check)

```bash
./Scripts/run_container.sh                                                          # interactive shell
./Scripts/run_container.sh 'cmake -S CUDA/Heterosplat/Source -B build && cmake --build build -j6'
./Scripts/run_container.sh 'python3 /heterosplat/Scripts/CaptureGsplatOracle.py /heterosplat/CUDA/Heterosplat/Source/UnitTests/Fixtures'
```

Container mounts the repo at `/heterosplat`. Build dir created in the container belongs to root on the host — wipe it from inside the container, not from the host.

### Current test count

34 tests (`./build/Check`), all passing. Suite layout:
- `Tensor.*` (10) — Core/Tensor.h
- `IntersectOffset.*` (3) — single-image, multi-image, zero-intersections
- `IntersectOffsetOracle.*` (1) — vs gsplat-Python
- `IntersectTile.*` (2) — dense AABB two-pass + packed image-id encoding
- `IntersectTileOracle.*` (1) — vs gsplat-Python, dense AABB fwd
- `ProjectionEWA3DGSFused.*` (3) — on-axis center, behind-camera cull, backward finite grads
- `ProjectionEWA3DGSFusedOracle.*` (1) — vs gsplat-Python, fwd
- `QuatScaleToCovar.*` (5) — closed-form forward × 3, closed-form backward, gradcheck backward
- `QuatScaleToCovarOracle.*` (2) — vs gsplat-Python, fwd + bwd
- `SphericalHarmonics.*` (4) — DC, single-basis, mask, gradcheck
- `SphericalHarmonicsOracle.*` (2) — vs gsplat-Python, fwd + bwd

## Adding the next kernel (the slice pattern)

Each new kernel from PLAN.md's Phase 0b table follows this pattern. The two done kernels (`quat_scale_to_covar`, `spherical_harmonics`) are reference implementations.

1. **Read upstream.** Open `repos/gsplat/gsplat/cuda/csrc/<Kernel>CUDA.cu`. Identify the `__global__` template (or templates) and the `launch_*` host wrapper.
2. **Vendor.** Create `CUDA/Heterosplat/Source/Kernels/Thirdparty/Gsplat/<Kernel>Kernels.cuh` with **only** the `__global__` templates and any `__device__` helpers they need, copied verbatim. Drop the `launch_*` and `AT_DISPATCH_*` blocks. If the kernel uses ATen-coupled helpers (e.g. `gpuAtomicAdd`), substitute the CUDA built-in (`atomicAdd`) and record the substitution in `NOTICE` and `VENDORED.md`.
3. **Launcher.** Create `CUDA/Heterosplat/Source/Kernels/Heterosplat/<Kernel>.{h,cu}` with `void launch_<canonical_op_name>_forward(...)` and `_backward(...)`. Keep gsplat's canonical operation name (e.g. `quat_scale_to_covar_preci`) for grep-ability; spell out `_forward` / `_backward` (per the project naming convention — see `MEMORY` if you have it). Doxygen the public API: param shapes, layouts, nullptr-skip semantics, write-vs-accumulate, math identity.
4. **Wire CMake.** Append the new `.cu` to `Source/Kernels/Heterosplat/CMakeLists.txt`'s `ADD_LIBRARY(HeterosplatKernels …)`.
5. **Closed-form / gradcheck tests.** Create `Source/UnitTests/Kernels/Heterosplat/<Kernel>_tests.cu`. At least one closed-form sanity test and one numerical gradcheck (centered finite-difference vs analytic backward). Pattern: see `QuatScaleToCovar_tests.cu` and `SphericalHarmonics_tests.cu`.
6. **Capture oracle fixture.** Add a `capture_<kernel>(out_dir, ...)` function to `Scripts/CaptureGsplatOracle.py` mirroring the existing two; run via `./Scripts/run_container.sh 'python3 /heterosplat/Scripts/CaptureGsplatOracle.py /heterosplat/CUDA/Heterosplat/Source/UnitTests/Fixtures'`.
7. **Oracle test.** Create `Source/UnitTests/Kernels/Heterosplat/<Kernel>_oracle_tests.cu` using `OracleFixture.h` to load fixtures and `expect_close` (`atol=1e-4, rtol=1e-4` covers nvcc-version drift). Pattern: see existing two oracle test files.
8. **Wire UnitTests CMake.** Append the new test files to `Source/UnitTests/CMakeLists.txt`'s `ADD_EXECUTABLE(Check …)`.
9. **LaTeX section.** Append a section to `Documents/LaTeX/KernelMathematics.tex`: source-file pointers → forward map → storage layout → launcher API correspondence (forward) → backward (VJP derivation) → launcher API correspondence (backward) → test correspondence. Match the structure of the existing two sections. Build with `latexmk -pdf KernelMathematics.tex`.
10. **Update PLAN.md status table** (Phase 0b sub-status) to flip the kernel's row.

The audit trail at any point: vendored kernels in `Thirdparty/Gsplat/` carry their upstream commit SHA in `VENDORED.md`; the LaTeX has source-file pointers per kernel; `Check`'s test names map directly to PLAN.md done-criteria.

## Where to start (next concrete action)

**Phase 0b kernel #6: `rasterize_to_pixels_3dgs`** (fwd + bwd). The final and heaviest kernel: tile rasterizer with per-tile cooperative groups and shared memory. After this, the `forward_backward_smoke_test` binary closes Phase 0b's done-criteria.

Open question worth flagging early for `rasterize_to_pixels_3dgs`: it uses tile-level cooperative groups and shared memory; check whether any helpers beyond `Common.h` / `Utils.cuh` need vendoring.

## Conventions

- **Git:** never commit/push to `master`/`main` — Ernest merges manually. Feature branches OK; `feat/` prefix.
- **C++ identifier style:** snake_case for class data members (with trailing `_` on private) and member/free functions; CamelCase for class/struct/file/directory names. Spell out names — no `numel`, `nbytes`, `cnt`, `idx`, `tmp`. Loop counters `i, j, k` in tight numerics are fine. **Exception:** if a math identifier has a single canonical long form (e.g. gsplat's `quat_scale_to_covar_preci`), keep that — vendoring conventions trump local rule, since the upstream-grep value is real.
- **Docs:** `PLAN.md` is the live plan. Update its "Status" table as phases complete. Don't write new planning files alongside it; amend. `Documents/LaTeX/KernelMathematics.tex` is the live math reference; append a section per new kernel.
- **Public ship discipline:** each phase ends with a real public artifact. Two consecutive Sundays without a ship → stop building, write up what exists.
- **No emojis in commit messages or markdown** unless explicitly requested.

## Pre-existing dev-env quirks (already fixed; don't re-discover)

These are baked into `Dockerfile.gsplat` already; just so you don't waste time:

- NGC PyTorch 26.02 has a libucs ABI mismatch (distro libucs missing `ucs_config_doc_nop`). Fixed via `ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH}`. The `ld.so.conf.d` approach loses to NGC's trusted-directory precedence at runtime — don't bother with it.
- `TORCH_CUDA_ARCH_LIST=8.6` (RTX 3060) and `MAX_JOBS=6` are pinned at image-build time.
- gsplat's CUDA extension is **pre-compiled at `docker build` time** via a final `RUN python -c "..."`, so first `import gsplat` inside a fresh container is sub-3s, not 14 minutes. We need this image alive in Phase 0b as the **numerical-correctness oracle** — capture gsplat-Python outputs once, store as test fixtures for our C++ launchers.

## Anti-patterns to avoid

- Don't reach for Python + gsplat because it's faster. The whole point is the C++/CUDA path. The Python tutorial-clone of this project would not be worth shipping for World Labs (their cofounder Mildenhall has seen every gsplat demo on Earth).
- Don't add a Rust orchestrator "for ergonomics." Single language is the whole point.
- Don't expand "heterogeneous" to >2 formats. The done-criteria is explicit.
- Don't write planning docs alongside `PLAN.md`. Amend `PLAN.md` instead.

## Open questions (deferred decisions)

See `PLAN.md` → "Open questions". Most relevant near-term: the Phase 2 second-source choice (default = (c) second photogrammetry capture with different coord convention). Decide before Phase 1 ships.
