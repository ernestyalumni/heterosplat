# Vendored gsplat headers (Apache-2.0). Header-only target — kernels are
# templated and instantiated by the launcher modules in
# ../../Heterosplat/. See ./NOTICE and ./VENDORED.md for provenance.

ADD_LIBRARY(GsplatVendored INTERFACE)

TARGET_INCLUDE_DIRECTORIES(GsplatVendored
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

# GSPLAT_BUILD_3DGS guards in upstream Config.h gate the kernels we use. We
# build only the 3DGS path; 2DGS / 3DGUT / lidar / external-distortion /
# camera-wrappers are out of scope.
TARGET_COMPILE_DEFINITIONS(GsplatVendored
  INTERFACE
  GSPLAT_BUILD_3DGS=1
  GSPLAT_BUILD_2DGS=0
  GSPLAT_BUILD_3DGUT=0
  GSPLAT_BUILD_ADAM=0
  GSPLAT_BUILD_RELOC=0
  )

TARGET_LINK_LIBRARIES(GsplatVendored INTERFACE glm::glm)
