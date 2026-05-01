################################################################################
## ColormapDebug — Phase 3 second custom CUDA kernel
################################################################################

ADD_LIBRARY(ColormapDebug
  ColormapDebug.cu
  )

TARGET_INCLUDE_DIRECTORIES(ColormapDebug PUBLIC ${CMAKE_SOURCE_DIR})

SET_TARGET_PROPERTIES(ColormapDebug
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

################################################################################
## Dear ImGui — immediate-mode GUI
################################################################################

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.8
  GIT_SHALLOW TRUE
  )
FetchContent_MakeAvailable(imgui)

ADD_LIBRARY(ImGui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )

TARGET_INCLUDE_DIRECTORIES(ImGui PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  )

TARGET_COMPILE_DEFINITIONS(ImGui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)

TARGET_LINK_LIBRARIES(ImGui PUBLIC glfw GLEW::GLEW)
