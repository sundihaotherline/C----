cmake_minimum_required(VERSION 3.10)
project(myfirstopencv)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
add_executable(CppTest main.cpp)

target_link_libraries(CppTest ${OpenCV_LIBS})
add_executable(color_picker main2.cpp)
target_link_libraries(color_picker ${OpenCV_LIBS}) 

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})