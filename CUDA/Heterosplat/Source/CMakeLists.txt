CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

# Single-source CUDA architecture: SM_86 (RTX 3060). Bumps to multi-arch
# happen when heterosplat needs to ship to other machines — Phase 4+.
SET(CMAKE_CUDA_ARCHITECTURES 86)

PROJECT(HETEROSPLAT LANGUAGES CXX CUDA)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR})

INCLUDE(CheckCXXCompilerFlag)

################################################################################
## Compiler standards — C++17 minimum (PLAN.md). Try C++20 first, fall back.
################################################################################

CHECK_CXX_COMPILER_FLAG(-std=c++20 COMPILER_SUPPORTS_CXX20)
CHECK_CXX_COMPILER_FLAG(-std=c++2a COMPILER_SUPPORTS_CXX2A)

IF(COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CXX_STANDARD "20")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
ELSEIF(COMPILER_SUPPORTS_CXX2A)
  SET(CMAKE_CXX_STANDARD "20")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
ELSE()
  SET(CMAKE_CXX_STANDARD "17")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()

MESSAGE("CMAKE_CXX_STANDARD: " ${CMAKE_CXX_STANDARD})

################################################################################
## CUDA toolkit
################################################################################

FIND_PACKAGE(CUDAToolkit REQUIRED)

IF(CUDAToolkit_FOUND)
  MESSAGE(STATUS "CUDA Toolkit found at ${CUDAToolkit_LIBRARY_DIR}")
ENDIF()
MESSAGE("CUDAToolkit_VERSION: " ${CUDAToolkit_VERSION})

# Ensure a CUDA C++20 standard option exists when the host toolchain claims
# C++20 support but CMake hasn't wired CUDA20.
IF(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  SET(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  SET(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
ENDIF()

IF(NOT (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0) AND COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CUDA_STANDARD 20)
  SET(CMAKE_CUDA_FLAGS "-std=c++20")
ENDIF()

MESSAGE("CMAKE_CUDA_STANDARD: " ${CMAKE_CUDA_STANDARD})

################################################################################
## ccache
################################################################################

FIND_PROGRAM(CCACHE_FOUND ccache)
IF(CCACHE_FOUND)
  SET(CMAKE_CXX_COMPILER_LAUNCHER ccache)
ENDIF()

################################################################################
## GLM — fetched at configure time. Upstream gsplat carries glm as a git
## submodule; we don't, to keep `git clone && cmake` working out of the box.
################################################################################

INCLUDE(FetchContent)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
  GIT_SHALLOW TRUE
  )
FetchContent_MakeAvailable(glm)

################################################################################
## GoogleTest — matches InServiceOfX/CUDALibraries/MoreCUDA convention.
################################################################################

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
FetchContent_MakeAvailable(googletest)

################################################################################
## Sub-modules
################################################################################

ADD_SUBDIRECTORY(Colmap)
ADD_SUBDIRECTORY(Core)
ADD_SUBDIRECTORY(Kernels)
ADD_SUBDIRECTORY(UnitTests)

################################################################################
## Forward-backward smoke test binary (Phase 0b done-criteria).
## Chains all 6 kernels + CUB glue into a single fwd+bwd pass on synthetic
## 1k-Gaussian data and verifies finite gradients.
################################################################################

ADD_EXECUTABLE(ForwardBackwardSmokeTest
  ForwardBackwardSmokeTest.cu
  )

TARGET_LINK_LIBRARIES(ForwardBackwardSmokeTest
  PRIVATE
  Core
  HeterosplatKernels
  )

SET_TARGET_PROPERTIES(ForwardBackwardSmokeTest
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
