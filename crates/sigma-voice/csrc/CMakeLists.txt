# Builds the vendored whisper.cpp as part of a parent project, so it is not "standalone": no
# examples, tests or files written into the source tree.
cmake_minimum_required(VERSION 3.14)
project(sigma_voice C CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp whisper.cpp)

set(SIGMA_LIBS whisper parakeet ggml ggml-base ggml-cpu)
if (TARGET ggml-blas)
    list(APPEND SIGMA_LIBS ggml-blas)
endif()
install(TARGETS ${SIGMA_LIBS} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
