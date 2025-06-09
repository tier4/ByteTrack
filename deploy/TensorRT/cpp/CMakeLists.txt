cmake_minimum_required(VERSION 2.6)

project(bytetrack)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Add OpenMP support
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/include/eigen3)
link_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
include_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/include)
link_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/lib64)
# tensorrt
include_directories(/opt/tiger/demo/TensorRT-7.2.3.4/include)
link_directories(/opt/tiger/demo/TensorRT-7.2.3.4/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


# compile bytetrack common files
set(BYTETRACK_COMMON_FILES
    ${PROJECT_SOURCE_DIR}/src/BYTETracker.cpp
    ${PROJECT_SOURCE_DIR}/src/kalmanFilter.cpp
    ${PROJECT_SOURCE_DIR}/src/lapjv.cpp
    ${PROJECT_SOURCE_DIR}/src/STrack.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
)
# compile bytetrack
add_executable(bytetrack
    ${PROJECT_SOURCE_DIR}/src/bytetrack.cpp
    ${BYTETRACK_COMMON_FILES}
)
target_link_libraries(bytetrack nvinfer)
target_link_libraries(bytetrack cudart)
target_link_libraries(bytetrack ${OpenCV_LIBS})


add_executable(bytetrack_comet 
    ${PROJECT_SOURCE_DIR}/src/bytetrack_comet.cpp
    ${BYTETRACK_COMMON_FILES}
)
target_link_libraries(bytetrack_comet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

