# Colmap — COLMAP .bin reader (no libcolmap dependency).

ADD_LIBRARY(Colmap
  ColmapReader.cpp
  )

TARGET_INCLUDE_DIRECTORIES(Colmap
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
