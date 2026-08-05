cmake_minimum_required(VERSION 3.15)
project(LaneTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Libraries
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

add_executable(lane_tracker main.cpp)

# Bind everything together
target_include_directories(lane_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(lane_tracker PRIVATE ${OpenCV_LIBRARIES} onnxruntime)

target_link_libraries(lane_tracker PRIVATE ${OpenCV_LIBRARIES} onnxruntime::onnxruntime)