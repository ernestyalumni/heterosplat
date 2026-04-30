# Core — owning device-buffer wrapper (Tensor.h, header-only) plus CUB
# device-level operation wrappers (CubOperations, compiled).

ADD_LIBRARY(Core
  CubOperations.cu
  )

TARGET_INCLUDE_DIRECTORIES(Core
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

SET_TARGET_PROPERTIES(Core
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
