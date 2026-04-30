# Training — optimizer kernels and loss functions.

ADD_LIBRARY(Training
  AdamOptimizer.cu
  ImageLoss.cu
  )

TARGET_INCLUDE_DIRECTORIES(Training
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

SET_TARGET_PROPERTIES(Training
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
