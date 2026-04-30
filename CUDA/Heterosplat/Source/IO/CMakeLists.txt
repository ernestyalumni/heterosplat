# IO — image loading (stb_image), PNG saving (stb_image_write), PLY output.

ADD_LIBRARY(IO
  ImageIO.cpp
  PlyWriter.cpp
  )

TARGET_INCLUDE_DIRECTORIES(IO
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_SOURCE_DIR}/../../../thirdparty/stb
  )
