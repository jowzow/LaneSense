cmake_minimum_required(VERSION 3.15)
project(LaneTracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Libraries
# Only the components main.cpp actually uses (VideoCapture, imshow/putText/line/cvtColor/
# resize). Pulling in the full OpenCV_LIBRARIES set links opencv_dnn, which statically
# registers its own copy of the ONNX operator schemas - duplicate registration alongside
# onnxruntime's own copy trips an assertion in onnx's SchemasRegisterer and the model
# fails to load. Scoping to these components keeps opencv_dnn out of the link entirely.
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui imgcodecs)
find_package(onnxruntime REQUIRED)

add_executable(lane_tracker main.cpp)

# Bind everything together
target_include_directories(lane_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(lane_tracker PRIVATE ${OpenCV_LIBS} onnxruntime::onnxruntime)