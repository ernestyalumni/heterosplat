# Training — optimizer kernels, loss functions, and activation utilities.

ADD_LIBRARY(Training
  Activations.cu
  AdamOptimizer.cu
  ImageLoss.cu
  )

TARGET_INCLUDE_DIRECTORIES(Training
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

SET_TARGET_PROPERTIES(Training
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
