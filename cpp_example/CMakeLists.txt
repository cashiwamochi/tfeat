cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(tfeat_cpp_demo)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/libtorch/)

find_package(Torch REQUIRED)

set(OpenCV_DIR ~/.local/OpenCV34/share/OpenCV)
find_package(OpenCV REQUIRED)
IF (OpenCV_FOUND)
  message([message] OpenCV_VER:${OpenCV_VERSION})
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIBRARY_DIRS})
  add_definitions(${OpenCV_DEFINITIONS})
ENDIF()

add_executable(
  tfeat_demo
  tfeat_demo.cpp
  )

target_link_libraries(
  tfeat_demo
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  )

set_property(
  TARGET tfeat_demo
  PROPERTY CXX_STANDARD 11)

