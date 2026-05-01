# heterosplat — Plan

> Fuse heterogeneous 3D sources into Gaussian Splat scenes + meshes. Custom CUDA kernels, automatic coordinate-frame normalization, and an interactive transform debugger.

Single-author project (Ernest Yeung) shipped publicly in phases. **Built in pure C++ + CUDA C++.** No Python, no PyTorch, no Rust. Single CMake build, single static binary. Vendors gsplat's CUDA kernels (Apache-2.0) and replaces their torch-thin launchers with raw-pointer launchers we own.

Primary purpose: portfolio evidence in the active World Labs interview cycle. Secondary purpose: a real, useful 3D-data tool.

Last updated: 2026-04-30.

---

## Why this project, why now

- **World Labs cycle is live.** Tech screen passed 2026-04-21. Next-round timing TBD; Monday 2026-04-27 follow-up to Nick Bell. Each interview round will likely surface "what have you been building?" — heterosplat is the answer.
- **Direct overlap with World Labs' stack.** Marble = 3D world generation + Gaussian splatting (Ben Mildenhall, co-founder). Pipeline Engineer (3D Data) role specifies: automated 3D data pipelines, synthetic data quality, GPU compute, shared tooling for developer velocity. Every clause of the README thesis maps to one of those bullets.
- **"Publish before apply"** — weekly QLA ritual (`COACHING_NOTES.md`, 20260424). Each phase ends with a public artifact a World Labs engineer could plausibly see.

## Why C++ + CUDA, no Python

- World Labs writes Python-primary + C++ for performance-critical 3D. Direct C++/CUDA chops is the more-direct signal.
- gsplat's `__global__` kernels are pure CUDA C++; only the launchers + ATen layer add a torch dependency. Vendoring + replacing those layers is straightforward.
- A Rust+CUDA story adds an FFI boundary without solving a real problem; pure C++ matches the production reality at most graphics-heavy shops.
- Single language, single build, single debugger.
- Every other gsplat-portfolio demo on Earth is Python. A clean C++/CUDA reimplementation that fuses heterogeneous sources is *not* tutorial-shaped.

## Architecture (target end state)

```
heterosplat/
├── CMakeLists.txt
├── src/
│   ├── colmap/                 COLMAP .bin/.txt loader (no libcolmap dep)
│   ├── normalize/              coord-frame normalization (pure C++)
│   ├── kernels/
│   │   ├── thirdparty/gsplat/  vendored .cu/.cuh (Apache-2.0)
│   │   └── heterosplat/        our custom kernels
│   ├── core/                   GPU buffer wrapper, training loop, Adam, losses
│   ├── viewer/                 (Phase 3) GLFW + ImGui + GL splat
│   └── cli/                    CLI11 front-end → single binary
├── thirdparty/                 single-header libs vendored in-tree
├── tests/                      GoogleTest unit + integration
└── data/                       small example fixtures
```

Output: one binary `heterosplat` with subcommands `train`, `render`, `view`, `normalize`.

## Dependencies

All single-header / vendored, no system installs needed beyond CUDA + CMake:
- **GLM** — math types (gsplat already uses it, CUDA-friendly).
- **stb_image / stb_image_write** — PNG IO.
- **nlohmann/json** — configs.
- **CLI11** — CLI parsing.
- **GoogleTest** — tests (FetchContent, matches `InServiceOfX/CUDALibraries/MoreCUDA` convention).
- **fmt** — formatting (vendored).
- **GLFW + Dear ImGui + glad** — Phase 3 viewer only.

System: CUDA Toolkit 13.x (already in `heterosplat:26.02-py3`), CMake 3.20+, C++17 compiler.

## Phasing

Each phase MUST end with a public artifact before the next begins. No "build everything then publish."

### Phase 0a — Vendor gsplat kernels, build them standalone

**Goal:** prove the libtorch-less build works.

**Done when:**
- gsplat's `csrc/*.cu` + `include/*.cuh` vendored under `src/kernels/thirdparty/gsplat/` (LICENSE + NOTICE preserved).
- Vendor only what 3DGS needs; explicitly skip 2DGS, 3DGUT, lidar, distortion, camera-wrappers.
- Our own `Tensor` type (~100 lines): owns `float* data` + `std::vector<int64_t> shape` + a stream — replaces `at::Tensor` in our launchers.
- A `cuda_smoke_test` binary that runs `quat_scale_to_covar_fwd` on synthetic input and prints correct 3×3 covariances.

**Public ship:** README "Build heterosplat in 30 seconds" with screenshot of `cuda_smoke_test`.

**Target:** ~3–5 days (by 2026-05-02).

### Phase 0b — Replace torch-thin launchers

**Goal:** every kernel needed for one fwd+bwd training step is callable from our C++ with raw pointers + shapes + stream.

**Done when:** all eight wrappers in `src/kernels/wrappers.cuh` taking only `(const float*, …, cudaStream_t)`:

| Kernel | Fwd | Bwd | Notes |
|---|---|---|---|
| `quat_scale_to_covar` | ✓ | ✓ | smallest; do this first |
| `spherical_harmonics` | ✓ | ✓ | math only |
| `intersect_tile` | ✓ | — | per-tile gaussian binning + sort |
| `intersect_offset` | ✓ | — | tile_offsets prefix sum |
| `projection_ewa_3dgs_fused` | ✓ | ✓ | 3D→2D EWA projection |
| `rasterize_to_pixels_3dgs` | ✓ | ✓ | the big tile rasterizer |

- GoogleTest covers each wrapper with a small numeric fixture vs. a Python+gsplat oracle (run inside the heterosplat docker container; capture outputs once, then check against them).
- A `forward_backward_smoke_test` binary runs fwd+bwd on a 1k-Gaussian, 64×64 image fixture and verifies output shapes + finite gradients.

**Target:** ~5–7 days (by 2026-05-09).

### Phase 1 — Train one scene end-to-end + first public demo

**Goal:** `heterosplat train --colmap path/to/scene` produces a `.ply` and an orbit-render mp4. Public artifact ships.

**Done when:**
- COLMAP loader for `cameras.bin` / `images.bin` / `points3D.bin` (formats are documented).
- Adam optimizer kernel (~50 lines CUDA).
- L1 + SSIM image-loss kernels (~150 lines combined).
- Training loop: COLMAP → init Gaussians from sparse points → ~30k iterations → save `.ply` (standard splat format readable by SuperSplat, antimatter15.com, etc.).
- `heterosplat render --ply X --orbit out.mp4` orbit-renders frames (stb_image_write) + stitches via `system("ffmpeg ...")`.
- Reproduction: README has `cmake -B build && cmake --build build && ./build/heterosplat train ...` working from a clean clone in the heterosplat docker image.

**Public ship:** X post or short blog post with the orbit video + 1-paragraph "what is this" + GitHub link. Tag gsplat/World-Labs-adjacent accounts.

**Out-of-scope for Phase 1:** custom CUDA kernels of our own, heterogeneous sources, viewer, 2DGS, mesh output.

**Target:** ~3 weeks total from project start (by 2026-05-19).

### Phase 2 — Heterogeneous fusion + coord-frame normalization + first custom CUDA kernel

**Goal:** ingest a second source format with different coordinate conventions, normalize into a shared frame, fuse with Phase 1's splat. Demonstrate failure-vs-fix.

**Decision needed before starting:** which second source. Default = **(c) second photogrammetry capture with different coordinate convention** (Y-up vs Z-up, metric vs unit-scale). Most visceral on video; normalization code generalizes.

**Done when:**
- `heterosplat normalize --source-a A.ply --source-b B.ply --convention auto` outputs a single normalized splat.
- Normalization library (`src/normalize/`) covers handedness flip, up-axis rotation, scale matching, origin centering — pure C++, unit-tested.
- **First custom CUDA kernel:** `apply_homogeneous_transform_kernel` — batched 4×4 transform applied to splat means + quaternion composition + scale chain rule. Reused by Phase 3 viewer for live re-rendering.
- Failure case rendered: un-normalized fusion (broken) vs normalized fusion (correct), side-by-side.

**Public ship:** blog post with side-by-side video. This is the post that matters for World Labs — pipeline thinking, not tutorial-running.

**Target:** ~3 weeks after Phase 1 (by 2026-06-09).

### Phase 3 — Interactive transform debugger + second custom kernel

**Goal:** GLFW + ImGui viewer. User adjusts a transform widget; splat re-renders live at >30 FPS.

**Done when:**
- GLFW window + ImGui controls for per-source translation/rotation/scale.
- Live rendering: each frame, `apply_homogeneous_transform_kernel` runs on the splat params, then our rasterizer renders to a CUDA buffer copied to an OpenGL texture (CUDA-GL interop, or staged via `cudaMemcpy` + `glTexSubImage2D`).
- **Second custom kernel** (choice depends on Phase 2 source pick): e.g. `depth_to_gaussian_seeds_kernel` (if depths) or `colormap_per_axis_debug_kernel` (for live visualization of frames).
- Recorded demo video.

**Public ship:** longer blog post + tweet thread + screen-recorded demo. Tag gsplat / World Labs accounts. This is the artifact most likely to get noticed.

**Out-of-scope:** web hosting, accounts, more than 2 source formats, mesh extraction.

**Target:** ~4 weeks after Phase 2 (by 2026-07-07).

## Done-criteria — what "heterogeneous" means concretely

- *Exactly two* input source types across the whole project:
  - Photogrammetry pipeline output (COLMAP) — Phase 1.
  - One of: mesh (.obj/.glb), depth-map sequence, or a second photogrammetry capture with different coordinate conventions — Phase 2.
- Anything beyond two formats is Phase 4+ and gated on the World Labs cycle resolving.

## Tech decisions (locked unless explicitly revisited)

| Decision | Choice | Why |
|---|---|---|
| Host language | C++17 | std::filesystem, std::optional, structured bindings |
| Device language | CUDA C++ | matches gsplat; nvcc native |
| Build | CMake 3.20+ | industry standard for CUDA |
| Math types | GLM | gsplat uses it; header-only; CUDA-friendly |
| Image IO | stb_image / stb_image_write | single-header |
| JSON | nlohmann/json | single-header |
| CLI | CLI11 | single-header |
| Tests | GoogleTest (FetchContent) | matches `InServiceOfX/CUDALibraries/MoreCUDA`; one `Check` exec, `gtest_discover_tests` |
| Viewer (P3) | GLFW + Dear ImGui + glad + CUDA-GL interop | all thin or single-header |
| GPU buffer | Custom ~100-line wrapper | no libtorch |
| Splat training kernels | Vendored gsplat, our launchers | algorithm is settled science; no value in rewriting |
| COLMAP | Direct .bin reader | documented format, no libcolmap dep |
| License | Apache 2.0 | matches vendored gsplat (update existing MIT LICENSE before first public ship) |
| Platform | Linux only initially | runs in heterosplat docker; macOS/Windows is Phase 4+ |

## Out-of-scope for the entire project (until World Labs cycle resolves)

- Python, PyTorch, Rust integration.
- Production deployment, hosting, accounts.
- More than 2 supported source formats.
- DCC plugins (Blender / Unity / Unreal).
- Mesh extraction.
- Multi-GPU training.
- 2DGS, 3DGUT, lidar, fisheye, distortion.
- Cross-platform (macOS / Windows).

## Public-ship cadence (QLA integration)

- **Sunday review** — every Sunday, 10 minutes:
  1. Did heterosplat ship a public increment this week?
  2. If no — what specifically blocked it, and what's the smallest commit-able increment by Thursday?
  3. Is heterosplat still the right primary World Labs evidence project, or has the cycle shifted (offer / rejection / new round)?
- **Anti-pattern guard:** if two consecutive Sundays show "no public ship," stop building and write up *whatever exists* as the artifact. A scoped public README beats a private 80%-done feature.

## Open questions

1. **Phase 2 source choice:** (a) mesh, (b) depth-maps, (c) second coordinate-convention capture. Default = (c). Decide before Phase 1 ships.
2. **Public hosting for demo videos:** GitHub README + X embeds, or `heterosplat.dev` page? Default = README + X.
3. **Naming:** "heterosplat" reads well in repo + on X. Confirm before first public ship.
4. **Phase 3 second custom kernel:** depends on Phase 2 source pick. Decide at end of Phase 2.
5. **License switch from MIT → Apache 2.0:** safer with vendored Apache-2.0 code. Default = switch before first public ship.
6. **Numerical-correctness oracle for Phase 0b tests:** capture gsplat-Python outputs once into fixture files, or run gsplat-Python at test time? Default = capture once (faster, deterministic, no torch dep at test runtime).

## Status

| Phase | Status | Public artifact |
|---|---|---|
| 0 — Dev environment | ✅ Done 2026-04-27 | Docker image with gsplat baked in |
| 0a — Vendor + standalone build | ✅ Done 2026-04-28 | CMake build, `Tensor` type, first vendored kernel + smoke test |
| 0b — Torch-free launchers | ✅ Done 2026-04-29 | All 6 kernels + CUB glue + ForwardBackwardSmokeTest (1k Gaussians, fwd+bwd) |
| 1 — Single-source train + render | 🔧 In progress | — |
| 2 — Heterogeneous + normalization + 1st custom kernel | ✅ Done 2026-04-30 | NormalizeAndFuse binary, LaTeX |
| 3 — Viewer + 2nd custom kernel | ✅ Done 2026-04-30 | SplatViewer binary, LaTeX |

### Phase 0b sub-status (2026-04-29)

| Kernel | Vendored | Launcher | Closed-form / gradcheck test | gsplat-Python oracle test |
|---|---|---|---|---|
| `quat_scale_to_covar`   | ✅ | ✅ fwd + bwd | ✅ | ✅ |
| `spherical_harmonics`   | ✅ | ✅ fwd + bwd | ✅ | ✅ |
| `intersect_tile`        | ✅ | ✅ fwd | ✅ AABB two-pass + packed | ✅ |
| `intersect_offset`      | ✅ | ✅ fwd | ✅ single/multi-image + zero | ✅ |
| `projection_ewa_3dgs_fused` | ✅ | ✅ fwd + bwd | ✅ | ✅ |
| `rasterize_to_pixels_3dgs`  | ✅ | ✅ fwd + bwd | ✅ | ✅ |

Total tests: 61 (`./build/Check`), all passing on RTX 3070 Laptop GPU (sm_86). LaTeX math reference (`Documents/LaTeX/KernelMathematics.tex`) covers all 6 kernels.

### Phase 2 sub-status (2026-04-30)

| Component | Status | Notes |
|---|---|---|
| Similarity transform kernel | ✅ Done | `apply_similarity_transform_kernel` — rotation + scale + translation on means/quats/log_scales |
| Homogeneous transform kernel | ✅ Done | `apply_homogeneous_transform_means_kernel` — 4x4 matrix on means with perspective division |
| Convention detection | ✅ Done | Up-axis auto-detection from point cloud statistics (variance heuristic) |
| Scene extent + normalization | ✅ Done | AABB extent, centroid, unit-sphere normalization |
| Y-up to Z-up rotation | ✅ Done | -90° around X axis |
| NormalizeAndFuse binary | ✅ Done | `--source-a A.ply [--source-b B.ply] --convention-a auto --output fused.ply` |
| PLY reader | ✅ Done (Phase 1) | Reads standard 3DGS format, auto-detects SH degree |
| Unit tests | ✅ Done | 13 new tests (7 Convention + 6 Transform), 78 total passing |
| LaTeX documentation | ✅ Done | `Documents/LaTeX/NormalizationMathematics.tex` (3 pages) |

Total tests: 78 (`./build/Check`), all passing.

### Phase 3 sub-status (2026-04-30)

| Component | Status | Notes |
|---|---|---|
| GLFW window + OpenGL 3.3 core | ✅ Done | System GLFW 3.3.6 + GLEW 2.2.0 |
| Dear ImGui integration | ✅ Done | FetchContent v1.91.8, GLFW+OpenGL3 backends |
| Orbit camera controls | ✅ Done | Left-drag orbit, right-drag pan, scroll zoom |
| Transform widget (ImGui) | ✅ Done | Translation/rotation/scale sliders + reset |
| Live similarity transform | ✅ Done | Per-frame copy + transform + activate |
| Full render pipeline per frame | ✅ Done | projection → SH → intersect → sort → rasterize |
| CUDA → GL texture display | ✅ Done | Download to host + glTexSubImage2D |
| Second custom kernel | ✅ Done | `colormap_per_axis_kernel` — turbo colormap of coordinate axes |
| Debug panel | ✅ Done | Colormap toggle + axis selector |
| Window resize handling | ✅ Done | Reallocates render buffers on framebuffer resize |
| Unit tests | ✅ Done | 6 new ColormapDebug tests, 84 total passing |
| LaTeX documentation | ✅ Done | `Documents/LaTeX/ViewerArchitecture.tex` (3 pages) |

Total tests: 84 (`./build/Check`), all passing.

`ForwardBackwardSmokeTest` binary chains all 6 kernels + CUB prefix sum + CUB radix sort into a single fwd+bwd pass on 1024 synthetic Gaussians (64x64 render, 16px tiles). Verifies finite non-zero gradients propagate through the full pipeline. CUB wrappers live in `Core/CubOperations.{h,cu}`.

### Phase 1 sub-status (2026-04-29)

| Component | Status | Notes |
|---|---|---|
| COLMAP binary reader | ✅ Done | cameras/images/points3D .bin, all 11 models |
| Image IO (stb) | ✅ Done | load_image + save_image_png |
| PLY writer | ✅ Done | Standard 3DGS format (SuperSplat/antimatter15) |
| Adam optimizer | ✅ Done | Per-element CUDA, bias-corrected moments |
| L1 loss | ✅ Done | Shared-memory reduction fwd + sign gradient bwd |
| Activation kernels | ✅ Done | sigmoid/exp fwd+bwd, quat normalize, view dirs |
| TrainSingleScene binary | ✅ Done | Full fwd+loss+bwd+Adam loop, SH degree schedule |
| SSIM loss | Not started | Deferred — L1 sufficient for Phase 1 |
| Orbit render | Not started | `--ply X --orbit out.mp4` with ffmpeg |
| CLI frontend (CLI11) | Not started | Currently argv-based |
| Densification | Not started | Out-of-scope for Phase 1 |

`TrainSingleScene` binary: `./TrainSingleScene <colmap_sparse> <images_dir> [output.ply] [iters]`. Loads COLMAP, initializes Gaussians from sparse points (SH DC from colors, isotropic log-scale from bounding-box spacing), runs 30k iterations of fwd+L1_loss+bwd+Adam with coarse-to-fine SH (degree bumps every 1000 iterations). Saves .ply checkpoints every 7000 iterations.

## Appendix — files referenced

- `repos/heterosplat/README.md` — project thesis.
- `repos/heterosplat/AGENTS.md` — entry point for AI agents (Claude / OpenClaw / Codex).
- `repos/heterosplat/Documents/LaTeX/KernelMathematics.tex` — math reference (one section per kernel).
- `repos/heterosplat/Documents/LaTeX/NormalizationMathematics.tex` — Phase 2 normalization math (similarity transforms, quaternion composition, convention detection).
- `repos/heterosplat/Documents/LaTeX/ViewerArchitecture.tex` — Phase 3 viewer pipeline, camera model, colormap kernel.
- `repos/heterosplat/Scripts/run_container.sh` / `run_tests.sh` / `CaptureGsplatOracle.py` — dev wrappers.
- `repos/heterosplat/CUDA/Heterosplat/Source/UnitTests/Fixtures/` — captured gsplat-Python oracle outputs (raw float32).
- `repos/Galvatron/Documents/WorldLabs/MASTER-PLAN.md` — interview cycle.
- `repos/Galvatron/Documents/WorldLabs/research/WORLD-LABS-CONTEXT.md` — what World Labs builds, what the roles want.
- `repos/Galvatron/Documents/Goals/COACHING_NOTES.md` — QLA priorities, weekly review ritual.
- `repos/InServiceOfX/Deployments/DockerContainers/Builds/Physics/Heterosplat/` — docker image build for this project.
- `repos/gsplat/gsplat/cuda/csrc/` — upstream kernel source we vendor.
- `repos/gsplat/gsplat/cuda/include/` — upstream headers we vendor.
- `repos/gsplat/gsplat/cuda/ext.cpp` — reference for the torch-coupled boundary we replace.
