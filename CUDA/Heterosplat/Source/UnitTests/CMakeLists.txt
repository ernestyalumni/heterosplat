ENABLE_TESTING()

ADD_EXECUTABLE(Check
  Colmap/ColmapReader_tests.cu
  Core/Tensor_tests.cu
  Kernels/Heterosplat/IntersectOffset_tests.cu
  Kernels/Heterosplat/IntersectOffset_oracle_tests.cu
  Kernels/Heterosplat/IntersectTile_tests.cu
  Kernels/Heterosplat/IntersectTile_oracle_tests.cu
  Kernels/Heterosplat/ProjectionEWA3DGSFused_tests.cu
  Kernels/Heterosplat/ProjectionEWA3DGSFused_oracle_tests.cu
  Kernels/Heterosplat/RasterizeToPixels3DGS_tests.cu
  Kernels/Heterosplat/RasterizeToPixels3DGS_oracle_tests.cu
  Kernels/Heterosplat/QuatScaleToCovar_tests.cu
  Kernels/Heterosplat/QuatScaleToCovar_oracle_tests.cu
  Kernels/Heterosplat/SphericalHarmonics_tests.cu
  Kernels/Heterosplat/SphericalHarmonics_oracle_tests.cu
  Training/Activations_tests.cu
  Training/AdamOptimizer_tests.cu
  Training/ImageLoss_tests.cu
  )

# Tests that load oracle fixtures need an absolute path to Fixtures/ baked
# in at build time -- so the binary works regardless of CWD or build dir.
TARGET_COMPILE_DEFINITIONS(Check
  PRIVATE
  HETEROSPLAT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Fixtures"
  )

TARGET_INCLUDE_DIRECTORIES(Check
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

TARGET_LINK_LIBRARIES(Check
  PRIVATE
  Colmap
  Core
  HeterosplatKernels
  Training
  gtest_main
  gmock_main
  )

INCLUDE(GoogleTest)
gtest_discover_tests(Check)

# Separable compilation lets us spread __global__ kernels across translation
# units; matches MoreCUDA's pattern.
SET_TARGET_PROPERTIES(Check
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(Check
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
