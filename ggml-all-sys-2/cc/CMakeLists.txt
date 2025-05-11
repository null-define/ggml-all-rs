cmake_minimum_required(VERSION 3.20) # for add_link_options and implicit target directories.

project(ggml-all-sys-2-cc)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/llama.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/whisper.cpp)

# todo
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/models/stable-diffusion.cpp)