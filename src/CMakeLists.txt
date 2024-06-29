# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc")

add_library(tinympc STATIC
    DeviceFunction.cu
    head.cu
)


target_include_directories(tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) 