# Core — minimal owning device-buffer wrapper. Header-only for now (Tensor
# is small enough to inline). Promoted to a compiled library later if more
# Core code lands.

ADD_LIBRARY(Core INTERFACE)

TARGET_INCLUDE_DIRECTORIES(Core
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
