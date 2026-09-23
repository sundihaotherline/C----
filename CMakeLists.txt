cmake_minimum_required(VERSION 3.10)
project(vision_training)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# 任务1（注意路径：src/task1_image/main1.cpp）
add_executable(task1_image src/task1_image/main1.cpp)
target_link_libraries(task1_image ${OpenCV_LIBS})

# 任务2（注意路径：src/task2_fit/main2.cpp）
add_executable(task2_fit src/task2_fit/main2.cpp)
target_link_libraries(task2_fit ${OpenCV_LIBS} Eigen3::Eigen)

# 如果之前那个 color_picker 还要用，就加上这句：
# add_executable(color_picker main4.cpp) 
# target_link_libraries(color_picker ${OpenCV_LIBS})