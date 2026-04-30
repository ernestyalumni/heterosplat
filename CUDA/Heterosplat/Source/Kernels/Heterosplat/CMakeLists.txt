# Heterosplat-owned launchers. Each kernel from gsplat's vendored set gets
# a raw-pointer (const float*, ..., cudaStream_t) launcher here. No at::Tensor.

ADD_LIBRARY(HeterosplatKernels
  IntersectOffset.cu
  IntersectTile.cu
  ProjectionEWA3DGSFused.cu
  QuatScaleToCovar.cu
  RasterizeToPixels3DGS.cu
  SphericalHarmonics.cu
  )

TARGET_INCLUDE_DIRECTORIES(HeterosplatKernels
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  )

TARGET_LINK_LIBRARIES(HeterosplatKernels
  PUBLIC
  GsplatVendored
  )

SET_TARGET_PROPERTIES(HeterosplatKernels
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
