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

# ONNX Runtime is vendored (third_party/onnxruntime_dml/, pulled from Microsoft's official
# Microsoft.ML.OnnxRuntime.DirectML + Microsoft.AI.DirectML NuGet packages) rather than
# taken from vcpkg. Two reasons: (1) vcpkg's onnxruntime port has no DirectML feature wired
# up, and (2) vcpkg's own onnxruntime build is separately broken (duplicate ONNX operator
# schema registration - the onnx port it depends on doesn't set
# ONNX_DISABLE_STATIC_REGISTRATION, so Ort::Session construction asserts on load). Vendoring
# a matched, official prebuilt set sidesteps both at once.
set(ONNXRUNTIME_DML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime_dml")

add_executable(lane_tracker main.cpp)

# Bind everything together
target_include_directories(lane_tracker PRIVATE ${OpenCV_INCLUDE_DIRS} "${ONNXRUNTIME_DML_DIR}/include")

target_link_libraries(lane_tracker PRIVATE ${OpenCV_LIBS} "${ONNXRUNTIME_DML_DIR}/lib/onnxruntime.lib")

# Vendored (not vcpkg-managed) runtime DLLs have to be copied next to the .exe by hand -
# VCPKG_APPLOCAL_DEPS only knows about vcpkg-tracked link targets, so it neither copies
# these nor (unlike the old vcpkg onnxruntime.dll) clobbers them on a rebuild.
add_custom_command(TARGET lane_tracker POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DML_DIR}/bin/onnxruntime.dll"
        "${ONNXRUNTIME_DML_DIR}/bin/onnxruntime_providers_shared.dll"
        "${ONNXRUNTIME_DML_DIR}/bin/DirectML.dll"
        "$<TARGET_FILE_DIR:lane_tracker>"
)