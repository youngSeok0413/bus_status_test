cmake_minimum_required(VERSION 3.10)
project(bus_status)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

add_executable(bus_status main.cpp)

target_link_libraries(bus_status ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})
